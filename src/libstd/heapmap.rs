// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! Heap allocation and GC mark-bit accounting data structures.
//!
//! Structures in this module are concerned with keeping track
//! of a large number of allocations: those covering the local
//! heap of each task.
//!
//! This module does not (at the time of this writing) contain
//! logic for efficiently locating free regions within a page
//! or moving values during GC. It could grow these capabilities
//! over time, but at the moment it assumes a lower-level allocator
//! such as malloc is _producing_ the allocations. This module just
//! contains structures to help index those allocations for the
//! sake of garbage collecting them.

#[deny(managed_heap_memory)];

use prelude::*;

use uint;
use trie::TrieMap;
use container::Map;
use option::{Some,None};
use libc;
use io::{fd_t,WriterUtil};

/// log_2(pagesize). On windows, pages are 64k. On most unixes they're
/// usually 4k. So these are 16 and 12 respectively.
///
/// Note: it is not presently necessary for page size to actually be
/// the size of an OS-page, though they are set that way as a first
/// approximation, and a few things may break if you pick a
/// non-power-of-2.
///
/// It is also _likely_ that we will keep these at OS page sizes
/// as we are likely to transition this map to managing the local
/// heap on its own, rather than layering on top of malloc. At
/// that point the size may become mandatory.
///
/// Synthetic tests suggest that it does not matter _that_ much
/// whether we pick 4k or 64k pages for this; real-world tests may
/// reveal a different answer.
#[cfg(target_os="win32")]
static BYTES_PER_PAGE_LOG2 : uint = 16;
#[not(cfg(target_os="win32"))]
static BYTES_PER_PAGE_LOG2 : uint = 12;

static BYTES_PER_PAGE : uint = 1 << BYTES_PER_PAGE_LOG2;
static PAGE_MASK : uint = BYTES_PER_PAGE - 1;

/// Page contents are tracked in terms of cells, not bytes. A cell is
/// the smallest unit you can get back from the underlying allocator
/// and/or natural unit of alignment for allocations. It is, in any
/// case, the smallest unit we track allocated-ness / free-ness in
/// terms of. This is set to 8 bytes presently, but it might need
/// per-platform or per-underlying-malloc adjustment.
static BYTES_PER_CELL_LOG2 : uint = 3;
static BYTES_PER_CELL : uint = 1 << BYTES_PER_CELL_LOG2;
static CELLS_PER_PAGE : uint = BYTES_PER_PAGE >> BYTES_PER_CELL_LOG2;

/// A cell can assume 4 states (see `CellState`) so it is mapped
/// in this structure using 2 bits.
static BITS_PER_CELL_MAP : uint = 2;

/// A cell can be in 4 states: `CELL_EMPTY`, `CELL_BEGIN_BLACK`,
/// `CELL_BEGIN_WHITE`, and `CELL_CONTD`. An allocation is represented
/// by one `BEGIN_*` cell followed by 0 or more `CELL_CONTD` cells.
/// Unallocated space is represented by `CELL_EMPTY`.
type CellState = u8;
static CELL_EMPTY : CellState = 0b00;
static CELL_BEGIN_WHITE : CellState = 0b01;
static CELL_BEGIN_BLACK : CellState = 0b10;
static CELL_CONTD : CellState = 0b11;
static CELL_MASK : uint = 0b11;

static BITS_PER_PAGE_MAP : uint = BITS_PER_CELL_MAP * CELLS_PER_PAGE;
static UINTS_PER_PAGE_MAP : uint = BITS_PER_PAGE_MAP / uint::bits;

static WORD_MASK : uint = uint::bits - 1;

#[cfg(target_word_size="32")]
static BITS_PER_WORD_LOG2 : uint = 5;

#[cfg(target_word_size="64")]
static BITS_PER_WORD_LOG2 : uint = 6;


/// Logical container for a large set of `(pos,len)` pairs, each
/// representing an allocation `[pos, pos+len)`. For each such
/// allocation, also maintains a mark bit that is set during GC to
/// either black (reachable) or white (unreachable).
///
/// Physically, a facade for two separate containers.
///
///   - `paged_allocs` maps page numbers to `PageMap`s, and contains
///     any allocation that fits within a page. _All_ the allocations
///     that occur on a given page are tracked by a _single_ `PageMap`
///     entry in `paged_allocs`.
///
///   - `extra_allocs` maps addresses to `HeapRecord`s directly.
///     _Each_ allocation stored this way is represented by a
///     _separate_ entry in `extra_allocs`.
///
/// This organization is a bet on space efficiency: we are betting
/// that size and spatial locality of allocations means many of them
/// cluster onto single pages and can be tracked by a smaller number
/// of overall bytes using `PageMap`s than if represented separately
/// as `HeapRecord`s. A `PageMap` is relatively dense under conditions
/// of high spatial locality and small allocations: Between (0.25 *
/// len) and 1024 bits per allocation, in the leaves of the page
/// trie. By contrast, `extra_allocs` costs between 256 and 4096
/// bits per allocation in the leaves of the allocation trie, with
/// many more interior nodes besides.
///
/// It is correct therefore to view `paged_allocs` as a
/// manually-compressed form of the same information that would be
/// stored in `extra_allocs`. Indeed, initial versions of the GC only
/// used a single `TrieMap` like `extra_allocs`, but it was very
/// space-costly.
///
/// The interface to `HeapMap` should present the operations that the
/// GC requires from a _single_ allocation index, and attempt to
/// service the operations by delegating to `paged_allocs` when
/// possible, `extra_allocs` only when necessary.

#[macro_escape]
mod zz {
    macro_rules! debug2 (($( $arg:expr ),+) => (()))
/*
    macro_rules! debug2 (
        ($( $arg:expr ),+) => {
            {
                let _e = libc::STDERR_FILENO as fd_t;
                _e.write_str(fmt!( $($arg),+ ));
                _e.write_str("\n");
            }
        }
    )
*/
}


pub struct HeapMap {
    priv len: uint,
    priv paged_allocs: TrieMap<~PageMap>,
    priv extra_allocs: TrieMap<HeapRecord>
}

pub impl HeapMap {
    fn new() -> HeapMap {
        HeapMap {
            len: 0,
            paged_allocs: TrieMap::new(),
            extra_allocs: TrieMap::new()
        }
    }

    fn page_num(pos: uint) -> uint {
        pos >> BYTES_PER_PAGE_LOG2
    }

    fn is_subpage_alloc(pos: uint, len: uint) -> bool {
        let len = PageMap::cell_count_of_len(len) * BYTES_PER_CELL;
        HeapMap::page_num(pos) == HeapMap::page_num(pos+len)
    }

    fn key_bounds(&self) -> (uint, uint) {
        let mut min = match self.extra_allocs.next(0) {
            None => 0,
            Some((k, _)) => k
        };
        let mut max = match self.extra_allocs.prev(uint::max_value) {
            None => uint::max_value,
            Some((k, r)) => k + r.size
        };

        min = match self.paged_allocs.next(0) {
            None => min,
            Some((k, _)) => min.min(&(k * BYTES_PER_PAGE))
        };
        max = match self.paged_allocs.prev(uint::max_value) {
            None => max,
            Some((k, _)) => max.max(&((k+1) * BYTES_PER_PAGE))
        };
        return (min, max);        
    }

    fn len(&self) -> uint { self.len }

    fn contains_key(&self, x: &uint) -> bool {
        self.extra_allocs.contains_key(x) ||
            self.paged_allocs_contains_key(x)
    }

    fn paged_allocs_contains_key(&self, x: &uint) -> bool {
        let page_num = HeapMap::page_num(*x);
        match self.paged_allocs.find(&page_num) {
            None => false,
            Some(ref pm) => pm.contains_alloc_starting_at_addr(*x)
        }
    }

    fn each_key(&mut self, f: &fn(&uint) -> bool) -> bool {
        self.extra_allocs.each_key(f);
            
        for self.paged_allocs.mutate_values |pagenum, pm| {
            let base = pagenum * BYTES_PER_PAGE;
            for pm.mutate_values(base) |pos, _, _| {
                if ! f(pos) {
                    return false;
                }
            };
        }
        return true;
    }

    fn mutate_values(&mut self, f: &fn(&uint, &mut HeapRecord) -> bool) -> bool {
        self.extra_allocs.mutate_values(f);
        for self.paged_allocs.mutate_values |pagenum, pm| {
            let base = pagenum * BYTES_PER_PAGE;
            for pm.mutate_values(base) |pos, len, is_marked| {
                let mut hr = HeapRecord {
                    size: *len,
                    is_marked: *is_marked
                };
                if ! f(pos, &mut hr) {
                    return false;
                }
                *is_marked = hr.is_marked;
            };
        }
        return true;
    }

    fn clear_marks_and_enumerate_unmarked(&mut self,
                                          f: &fn(uint,uint) -> bool) {
        for self.mutate_values |pos, hr| {
            if hr.is_marked {
                if !f(*pos, hr.size) {
                    return;
                }
            }
            hr.is_marked = false;
        }
    }


    fn mutate_prev(&mut self, addr: uint, f: &fn(uint, &mut HeapRecord)) {
        // Pointer could be anything, we only call back if we find it.
        let page_num = HeapMap::page_num(addr);
        match self.paged_allocs.find_mut(&page_num) {
            Some(pm) => {
                let (state, pos, len) = pm.find_alloc_pos_len(addr);
                assert!(state != CELL_CONTD);
                if state != CELL_EMPTY {
                    let marked = state == CELL_BEGIN_BLACK;
                    let mut hr = HeapRecord {
                        size: len,
                        is_marked: marked
                    };
                    f(pos, &mut hr);
                    if hr.is_marked != marked {
                        pm.set_cell(PageMap::cell_of_addr(pos),
                                    if hr.is_marked {
                                        CELL_BEGIN_BLACK
                                    } else {
                                        CELL_BEGIN_WHITE
                                    });
                    }
                    return;
                }
            }
            None => ()
        }

        // Didn't find it in the page_map, try the trie map.
        self.extra_allocs.mutate_prev(addr, f);
    }

    fn insert(&mut self, pos: uint, hr: HeapRecord) -> bool {
        assert!(!hr.size != 0);
        let mut was_new = false;
        if HeapMap::is_subpage_alloc(pos, hr.size) {
            let page_num = HeapMap::page_num(pos);
            debug2!("inserting 0x%x + %u into page map at page 0x%x (of 0x%x), offset 0x%x",
                   pos, hr.size, page_num, self.paged_allocs.len(), pos & PAGE_MASK);
            let mut hit = false;
            match self.paged_allocs.find_mut(&page_num) {
                Some(pm) => {
                    debug2!("found existing page 0x%x", page_num);
                    was_new = pm.add_alloc(pos, hr.size, hr.is_marked);
                    // pm.self_check();
                    hit = true;
                }
                None => ()
            }
            if hit {
                self.len += 1;
                return was_new;
            }
            debug2!("making new page 0x%x", page_num);
            let mut pm = ~PageMap::new();
            self.len += 1;
            pm.add_alloc(pos, hr.size, hr.is_marked);
            // pm.self_check();
            was_new = self.paged_allocs.insert(page_num, pm);
            assert!(was_new);
        } else {
            debug2!("inserting 0x%x + %u into trie map",
                   pos, hr.size);
            was_new = self.extra_allocs.insert(pos, hr);
            self.len += 1;
        }
        return was_new;
    }

    fn remove(&mut self, pos: &uint) -> bool {
        let mut found = false;
        let pos = *pos;
        assert!(self.len != 0);
        if self.paged_allocs_contains_key(&pos) {
            let page_num = HeapMap::page_num(pos);
            debug2!("removing 0x%x from page map at page 0x%x, offset 0x%x",
                   pos, page_num, pos & PAGE_MASK);
            let mut delete_node = false;
            match self.paged_allocs.find_mut(&page_num) {
                Some(pm) => {
                    let (state, pos, len) = pm.find_alloc_pos_len(pos);
                    debug2!("found alloc pos=%u, len=%u",
                           pos, len);
                    assert!(state == CELL_BEGIN_WHITE ||
                            state == CELL_BEGIN_BLACK);
                    found = pm.clear_alloc(pos, len);
                    // pm.self_check();
                    delete_node = pm.len == 0;
                }
                None => ()
            }
            if delete_node {
                debug2!("removing empty page 0x%x", page_num);
                self.paged_allocs.remove(&page_num);
            }
        } else {
            debug2!("removing 0x%x from trie map", pos);
            found = self.extra_allocs.remove(&pos);
        }
        self.len -= 1;
        return found;
    }

}


/// Helper structure stored within HeapMap and conveyed to clients
/// when inspecting and allocation records and their mark bit.
pub struct HeapRecord {
    size: uint,
    is_marked: bool
}


/// PageMap maps the allocations residing in a single page of the
/// heap. It is a contiguous array of words but treated logically as
/// an array of 2-bit values, each mapping a single "cell" (likely 8
/// bytes). See `BYTES_PER_CELL_LOG2` and `CellState`.
struct PageMap {
    len: uint,
    words: [uint, ..UINTS_PER_PAGE_MAP]
}

impl PageMap {

    fn new() -> PageMap {
        PageMap {
            len: 0,
            words: [0, ..UINTS_PER_PAGE_MAP ]
        }
    }

    #[inline(always)]
    fn cell_of_addr(addr: uint) -> uint {
        ((addr & PAGE_MASK) >> BYTES_PER_CELL_LOG2) as uint
    }

    #[inline(always)]
    fn cell_count_of_len(len: uint) -> uint {
        let mask = BYTES_PER_CELL - 1;
        ((len + mask) & !mask) >> BYTES_PER_CELL_LOG2
    }

    #[inline(always)]
    fn cell_word_and_bit_offset(cell: uint) -> (uint,uint) {
        let bit = cell * BITS_PER_CELL_MAP;
        (bit >> BITS_PER_WORD_LOG2, bit & WORD_MASK)
    }

    fn word_to_cells_str(x: uint) -> ~str {
        use str::OwnedStr;
        let mut s = ~"[";
        let mut bit = uint::bits;
        while bit != 0 {
            if bit != uint::bits {
                s.push_char('|');
            }
            for uint::range(0, BITS_PER_CELL_MAP) |_| {
                bit -= 1;
                if (x >> bit) & 1 == 1 {
                    s.push_char('1');
                } else {
                    s.push_char('0');
                }
            }
        }
        s.push_char(']');
        s
    }

    fn self_check(&self) {
        let mut prev_state = CELL_EMPTY;
        for uint::range(0, CELLS_PER_PAGE) |cell| {
            let state = self.get_cell(cell);
            match (prev_state, state) {

                // Empty cells and allocation-beginnings are ok
                (CELL_EMPTY, CELL_BEGIN_BLACK) |
                (CELL_EMPTY, CELL_BEGIN_WHITE) |
                (CELL_EMPTY, CELL_EMPTY) => (),

                // One-cell allocations are ok.
                (CELL_BEGIN_BLACK, CELL_EMPTY) |
                (CELL_BEGIN_WHITE, CELL_EMPTY) |

                // Multi-cell allocations are ok
                (CELL_BEGIN_BLACK, CELL_CONTD) |
                (CELL_BEGIN_WHITE, CELL_CONTD) |
                (CELL_CONTD, CELL_CONTD) |
                (CELL_CONTD, CELL_EMPTY) |
                (CELL_CONTD, CELL_BEGIN_BLACK) |
                (CELL_CONTD, CELL_BEGIN_WHITE) => (),

                // Everything else is wrong
                (a,b) => {
                    fail!("heapmap self-check failed: (0x%x,0x%x)",
                          a as uint, b as uint);
                }
            }
            prev_state = state;
        }
    }

    #[inline(always)]
    fn get_cell(&self, cell: uint) -> CellState {
        let (w, b) = PageMap::cell_word_and_bit_offset(cell);
        let state = ((self.words[w] >> b) & CELL_MASK) as CellState;
        // debug2!("get cell %u: (word %u, bit %u) = %u", cell, w, b, state as uint);
        // debug2!("word: %s", PageMap::word_to_cells_str(self.words[w]));
        return state;
    }

    #[inline(always)]
    fn set_cell(&mut self, cell: uint, state: CellState) {
        let (w, b) = PageMap::cell_word_and_bit_offset(cell);
        // debug2!("set cell %u: (word %u, bit %u) = %u", cell, w, b, state as uint);
        // debug2!("word -: %s", PageMap::word_to_cells_str(self.words[w]));
        self.words[w] = ((self.words[w] & !(CELL_MASK << b)) |
                         ((state as uint) << b));
        // debug2!("word +: %s", PageMap::word_to_cells_str(self.words[w]));
    }

    fn contains_alloc_starting_at_addr(&self, addr: uint) -> bool {
        let cell_num = PageMap::cell_of_addr(addr);
        let CellState = self.get_cell(cell_num);
        return CellState == CELL_BEGIN_WHITE ||
            CellState == CELL_BEGIN_BLACK;
    }

    /// Returns (CELL_EMPTY, addr, BYTES_PER_CELL) if there's no such
    /// allocation, otherwise the (CELL_BEGIN_{WHITE,BLACK}, pos, len)
    /// of the spanning allocation.

    fn find_alloc_pos_len(&self, addr: uint) -> (CellState, uint, uint) {

        let mut pos = addr;
        let mut len = BYTES_PER_CELL;

        let mut lo_cell = PageMap::cell_of_addr(addr);
        let mut hi_cell = lo_cell;

        let mut lo_state = self.get_cell(lo_cell);
        let mut hi_state = lo_state;

        while lo_state == CELL_CONTD && lo_cell > 0 {
            pos -= BYTES_PER_CELL;
            len += BYTES_PER_CELL;
            lo_cell -= 1;
            lo_state = self.get_cell(lo_cell);
        }

        assert!(lo_state != CELL_CONTD);

        while (hi_cell+1) < CELLS_PER_PAGE {
            hi_cell += 1;
            hi_state = self.get_cell(hi_cell);
            if hi_state != CELL_CONTD {
                break;
            }
            len += BYTES_PER_CELL;
        }

        assert!(hi_state != CELL_CONTD);

        return (lo_state, pos, len);
    }

    // XXX: at the moment we only mutate marks, not length.
    // but there should not be much reason for the latter.
    fn mutate_values(&mut self, mut addr: uint,
                     f: &fn(&uint, &uint, &mut bool) -> bool) -> bool {
        let mut cell = 0;
        while cell < CELLS_PER_PAGE {
            let state = self.get_cell(cell);

            if state == CELL_BEGIN_BLACK || state == CELL_BEGIN_WHITE {
                let pos = addr;
                let cell_begin = cell;
                let mut len = BYTES_PER_CELL;
                let mut is_marked = state == CELL_BEGIN_BLACK;
                let mark_begin = is_marked;
                while cell+1 < CELLS_PER_PAGE && 
                    self.get_cell(cell+1) == CELL_CONTD {
                    addr += BYTES_PER_CELL;
                    len += BYTES_PER_CELL;
                    cell += 1;
                }

                if !f(&pos, &len, &mut is_marked) {
                    return false;
                }

                if is_marked != mark_begin {
                    self.set_cell(cell_begin,
                                  if is_marked {
                                      CELL_BEGIN_BLACK
                                  } else {
                                      CELL_BEGIN_WHITE
                                  });
                }

            } else if state == CELL_CONTD {
                fail!("unexpected isolated CELL_CONTD");

            } else {
                // CELL_EMPTY is ignorable.
                ()
            }
                    
            addr += BYTES_PER_CELL;
            cell += 1;
        }
        true
    }

    #[inline(always)]
    fn set_pos_and_len(&mut self, pos: uint, len: uint,
                       first: CellState, rest: CellState) {
        let mut cell = PageMap::cell_of_addr(pos);
        let mut state = first;
        debug2!("setting pos+len for %u cells", 
               PageMap::cell_count_of_len(len));
        for PageMap::cell_count_of_len(len).times {
            self.set_cell(cell, state);
            cell += 1;
            state = rest;
        }
    }

    fn clear_alloc(&mut self, pos: uint, len: uint) -> bool {
        let cell_num = PageMap::cell_of_addr(pos);
        let was_nonempty = self.get_cell(cell_num) != CELL_EMPTY;
        self.set_pos_and_len(pos, len, CELL_EMPTY, CELL_EMPTY);
        self.len -= 1;
        return was_nonempty;
    }

    fn add_alloc(&mut self, pos: uint, len: uint, is_marked: bool) -> bool {
        let cell_num = PageMap::cell_of_addr(pos);
        let was_empty = self.get_cell(cell_num) == CELL_EMPTY;
        let begin = if is_marked { CELL_BEGIN_BLACK } else { CELL_BEGIN_WHITE };
        self.set_pos_and_len(pos, len, begin, CELL_CONTD);
        self.len += 1;
        return was_empty;
    }
}

#[cfg(test)]
mod test {

    use HeapMap;
    use core::uint;
    use BYTES_PER_CELL;
    use BYTES_PER_PAGE;

    fn check_insert_range(hm: &mut HeapMap, lo: uint, hi: uint,
                          step: uint, sz: uint) {
        let mut len = hm.len();
        for uint::range_step(lo, hi, step as int) |i| {
            debug2!("check_insert_range(lo=%u, hi=%u, step=%u, sz=%u) i=%u",
                   lo, hi, step, sz, i);
            assert!(! hm.contains_key(&i));
            hm.insert(i, HeapRecord { size: sz, is_marked: false });
            len += 1;
            assert!(hm.contains_key(&i));
            assert!(hm.len() == len);
        }
    }

    fn check_remove_range(hm: &mut HeapMap, lo: uint, hi: uint,
                          step: uint) {
        let mut len = hm.len();
        for uint::range_step(lo, hi, step as int) |i| {
            debug2!("check_remove_range(lo=%u, hi=%u, step=%u) i=%u",
                   lo, hi, step, i);
            assert!(hm.contains_key(&i));
            hm.remove(&i);
            len -= 1;
            assert!(!hm.contains_key(&i));
            assert!(hm.len() == len);
        }
    }

    static sizes : &'static [uint] = &[1, 2, 3, 4, 5,
                                       7, 8, 9,
                                       15, 16, 17,
                                       31, 32, 33,
                                       62, 64, 65,
                                       255, 256, 257];
    static gaps : &'static [uint] = &[1, 2, 3, 4, 8];
    
    #[test]
    fn paged_1_page_seq_insert_and_delete() {
        let mut hm = HeapMap::new();
        let sz = sizes.map(|&x| x * BYTES_PER_CELL);
        for sz.each |&sz| {
            for gaps.each |&gap| {
                check_insert_range(&mut hm, 0, 1000, sz+gap, sz);
                assert!(hm.paged_allocs.len() == 1);
                check_remove_range(&mut hm, 0, 1000, sz+gap);
                assert!(hm.paged_allocs.len() == 0);
                assert!(hm.len() == 0);

                check_insert_range(&mut hm, 0, 1000, sz+gap, sz);
                assert!(hm.paged_allocs.len() == 1);
                check_remove_range(&mut hm, 0, 1000, sz+gap);
                assert!(hm.paged_allocs.len() == 0);
                assert!(hm.len() == 0);
            }
        }
        assert!(hm.paged_allocs.len() == 0);
        assert!(hm.len() == 0);
    }

    #[test]
    fn paged_multipage_seq_insert_and_delete() {
        for 100.times {
        let mut hm = HeapMap::new();
        let sz = sizes.map(|&x| x * BYTES_PER_CELL);
        for sz.each |&sz| {
            for gaps.each |&gap| {
                for uint::range(0, 20) |page| {
                    let lo = BYTES_PER_PAGE * page;
                    let hi = lo + 1000;
                    check_insert_range(&mut hm, lo, hi, sz+gap, sz);
                }
                for uint::range(0, 20) |page| {
                    let lo = BYTES_PER_PAGE * page;
                    let hi = lo + 1000;
                    check_remove_range(&mut hm, lo, hi, sz+gap);
                }
                assert!(hm.len() == 0);
            }
        }
        assert!(hm.len() == 0);
        }
    }

    #[test]
    fn trie_multipage_seq_insert_and_delete() {
        for 100.times {
        let mut hm = HeapMap::new();
        let sz = sizes.map(|&x| x * BYTES_PER_CELL * 10);
        for sz.each |&sz| {
            for gaps.each |&gap| {
                for uint::range(0, 20) |page| {
                    let lo = BYTES_PER_PAGE * page * 10;
                    let hi = lo + 10000;
                    check_insert_range(&mut hm, lo, hi, sz+gap, sz);
                }
                for uint::range(0, 20) |page| {
                    let lo = BYTES_PER_PAGE * page * 10;
                    let hi = lo + 10000;
                    check_remove_range(&mut hm, lo, hi, sz+gap);
                }
                assert!(hm.len() == 0);
            }
        }
        assert!(hm.len() == 0);
        }
    }
}