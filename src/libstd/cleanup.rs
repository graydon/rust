// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)];
//#[deny(managed_heap_memory)];

use cast::transmute;
#[cfg(not(test))] use unstable::lang::clear_task_borrow_list;
use io::{fd_t, WriterUtil};
use libc::{c_char, c_void, c_uint, intptr_t, uintptr_t};
use libc;
use managed::raw::{BoxRepr, RC_MANAGED_UNIQUE, RC_IMMORTAL};
use option::{Some,None};
use ptr::{mut_null, null, to_unsafe_ptr};
use sys::size_of;
use sys::{TypeDesc, size_of};
use task::each_retained_ptr;
use uint;
use vec;
use vec::OwnedVector;
use i64;
use rt::global_heap;
use unstable::intrinsics::ctlz64;
use old_iter::BaseIter;

use container::{Container, Mutable, Map};
use trie::{TrieMap, TrieSet};

#[cfg(not(test))] use ptr::to_unsafe_ptr;

/**
 * Runtime structures
 *
 * NB: These must match the representation in the C++ runtime.
 */

type DropGlue<'self> = &'self fn(**TypeDesc, *c_void);
type FreeGlue<'self> = &'self fn(**TypeDesc, *c_void);

type TaskID = uintptr_t;

#[cfg(target_word_size = "64")]
pub struct StackSegment {
    prev: *StackSegment,
    next: *StackSegment,
    end: uintptr_t,
    valgrind_id: c_uint,
    rust_task: *Task,
    canary: uintptr_t,
    data: u8
}

#[cfg(target_word_size = "32")]
pub struct StackSegment {
    prev: *StackSegment,
    next: *StackSegment,
    end: uintptr_t,
    valgrind_id: c_uint,
    pad: u32,
    rust_task: *Task,
    canary: uintptr_t,
    data: u8
}

struct Scheduler { priv opaque: () }
struct SchedulerLoop { priv opaque: () }
struct Kernel { priv opaque: () }
struct Env { priv opaque: () }
struct AllocHeader { priv opaque: () }
struct MemoryRegion { priv opaque: () }

#[cfg(target_arch="x86")]
struct Registers {
    data: [u32, ..16]
}

#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
struct Registers {
    data: [u32, ..32]
}

#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
struct Context {
    regs: Registers,
    next: *Context,
    pad: [u32, ..3]
}

#[cfg(target_arch="x86_64")]
struct Registers {
    data: [u64, ..22]
}

#[cfg(target_arch="x86_64")]
struct Context {
    regs: Registers,
    next: *Context,
    pad: uintptr_t
}

pub struct BoxedRegion {
    env: *Env,
    backing_region: *MemoryRegion,
    live_allocs: *BoxRepr,
}

#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
pub struct Task {
    // Public fields
    refcount: intptr_t,                 // 0
    id: TaskID,                         // 4
    pad: [u32, ..2],                    // 8
    ctx: Context,                       // 16
    stack_segment: *StackSegment,       // 96
    runtime_sp: uintptr_t,              // 100
    scheduler: *Scheduler,              // 104
    scheduler_loop: *SchedulerLoop,     // 108

    // Fields known only to the runtime
    kernel: *Kernel,                    // 112
    name: *c_char,                      // 116
    list_index: i32,                    // 120
    boxed_region: BoxedRegion,          // 128
    gc: *c_void,                        // 132
}

#[cfg(target_arch="x86_64")]
pub struct Task {
    // Public fields
    refcount: intptr_t,
    id: TaskID,
    ctx: Context,
    stack_segment: *StackSegment,
    runtime_sp: uintptr_t,
    scheduler: *Scheduler,
    scheduler_loop: *SchedulerLoop,

    // Fields known only to the runtime
    kernel: *Kernel,
    name: *c_char,
    list_index: i32,
    boxed_region: BoxedRegion,
    gc: *c_void,
}

// A transmuted ~to one of these is held in the task.
// The annihilator drops it. Until then, all GC runs
// through it.

struct HeapRecord {
    size: uint,
    is_marked: bool
}

pub struct Gc {
    task: *Task,
    debug_gc: bool,
    actually_gc: bool,
    gc_zeal: bool,
    report_gc_stats: bool,
    check_gc_consistency: bool,

    phase: GcPhase,
    free_buffer: TrieSet,
    precious: TrieSet,
    precious_freed: TrieSet,
    tls_marked: TrieSet,
    heap: TrieMap<HeapRecord>,
    lowest: uint,
    highest: uint,
    marking_stack: ~[(uint,uint)],

    threshold: uint,
    alloc_count: uint,

    n_collections: uint,
    n_ns_marking: Stat,
    n_ns_sweeping: Stat,
    n_ns_total: Stat,
    n_mallocs: Stat,
    n_reallocs: Stat,
    n_explicit_frees: Stat,
    n_boxes_marked_from_stack: Stat,
    n_bytes_marked_from_stack: Stat,
    n_boxes_marked_from_tls: Stat,
    n_bytes_marked_from_tls: Stat,
    n_boxes_swept: Stat,
    n_bytes_swept: Stat,
    n_boxes_annihilated: Stat,
    n_bytes_annihilated: Stat,
    n_boxes_skipped_uniq: Stat,
    n_boxes_skipped_free: Stat,
}

#[cfg(unix)]
fn debug_mem() -> bool {
    ::rt::env::get().debug_mem
}

#[deriving(Eq)]
enum GcPhase {
    GcIdle,
    GcStarting,
    GcMarkingStack,
    GcMarkingTls,
    GcSweeping,
    GcAnnihilating
}

#[cfg(stage0)]
static always_gc : bool = false;

#[cfg(stage1)]
#[cfg(stage2)]
#[cfg(stage3)]
static always_gc : bool = true;

#[cfg(unix)]
fn check_flag(f: &str) -> bool {
    use libc;
    use os;
    use ptr::null;
    do os::as_c_charp(f) |p| {
        unsafe { libc::getenv(p) != null() }
    }
}

#[cfg(windows)]
fn check_flag(f: &str) -> bool {
    false;
}

pub impl Gc {
    fn get_task_gc() -> &mut Gc {
        unsafe {
            let tp : *Task = transmute(rust_get_task());
            let task : &mut Task = transmute(tp);
            if task.gc == null() {
                task.gc = transmute(~Gc::new(tp));
            }
            let p: &mut ~Gc = transmute(&(task.gc));
            &mut (**p)
        }
    }

    fn new(t: *Task) -> Gc {

        let zealous = check_flag("RUST_GC_ZEAL");
        let dbg = check_flag("RUST_DEBUG_GC");

        Gc {
            task: t,
            debug_gc: dbg,
            actually_gc: !check_flag("RUST_INHIBIT_GC") &&
                (always_gc ||
                 zealous ||
                 check_flag("RUST_ACTUALLY_GC")),
            gc_zeal: zealous,
            report_gc_stats: dbg ||
                check_flag("RUST_REPORT_GC_STATS"),
            check_gc_consistency: zealous ||
                dbg ||
                check_flag("RUST_CHECK_GC_CONSISTENCY"),

            phase: GcIdle,
            free_buffer: TrieSet::new(),
            precious:  TrieSet::new(),
            precious_freed:  TrieSet::new(),
            tls_marked:  TrieSet::new(),
            heap: TrieMap::new(),
            lowest: 0,
            highest: 0,
            marking_stack: ~[],

            threshold: 1024,
            alloc_count: 0,

            n_collections: 0,
            n_ns_marking: Stat::new(),
            n_ns_sweeping: Stat::new(),
            n_ns_total: Stat::new(),
            n_mallocs: Stat::new(),
            n_reallocs: Stat::new(),
            n_explicit_frees: Stat::new(),
            n_boxes_marked_from_tls: Stat::new(),
            n_bytes_marked_from_tls: Stat::new(),
            n_boxes_marked_from_stack: Stat::new(),
            n_bytes_marked_from_stack: Stat::new(),
            n_boxes_swept: Stat::new(),
            n_bytes_swept: Stat::new(),
            n_boxes_annihilated: Stat::new(),
            n_bytes_annihilated: Stat::new(),
            n_boxes_skipped_uniq: Stat::new(),
            n_boxes_skipped_free: Stat::new(),
        }
    }

    #[inline(always)]
    fn stderr_fd() -> fd_t {
        libc::STDERR_FILENO as fd_t
    }

    // Note: this will allocate several @-boxes
    unsafe fn debug_opaque_box(addr: uint) {
        use repr;
        use reflect;
        use intrinsic::{TyDesc, TyVisitor, visit_tydesc};
        use io;

        let box: &BoxRepr = transmute(addr);
        let tydesc: *TyDesc = transmute(box.header.type_desc);
        let writer = io::fd_writer(Gc::stderr_fd(), false);

        let mut u = repr::ReprVisitor(transmute(&box.data),
                                      writer);
        let v = reflect::MovePtrAdaptor(u);
        visit_tydesc(tydesc, @v as @TyVisitor)
    }

    #[inline(always)]
    fn debug_str(&self, s: &str) {
        if self.debug_gc {
            Gc::stderr_fd().write_str(s)
        }
    }

    #[inline(always)]
    fn debug_uint(&self, n: uint) {
        if self.debug_gc {
            Gc::stderr_fd().write_uint(n)
        }
    }

    #[inline(always)]
    fn debug_str_hex(&self, s: &str, p: uint) {
        if self.debug_gc {
            Gc::write_str_hex(s, p);
        }
    }

    #[inline(always)]
    fn debug_str_range(&self, s: &str,
                       p: uint, len: uint) {
        if self.debug_gc {
            let e = Gc::stderr_fd();
            e.write_str(s);
            e.write_str(": [");
            e.write_hex_uint(p);
            e.write_str(", ");
            e.write_hex_uint(p + len);
            e.write_str(") = ");
            e.write_uint(len);
            e.write_str(" bytes\n");
        }
    }

    fn write_str_uint(s: &str, n: uint) {
        let e = Gc::stderr_fd();
        e.write_str(s);
        e.write_str(": ");
        e.write_uint(n);
        e.write_str("\n");
    }

    fn write_str_hex(s: &str, n: uint) {
        let e = Gc::stderr_fd();
        e.write_str(s);
        e.write_str(": ");
        e.write_hex_uint(n);
        e.write_str("\n");
    }

    fn flush_and_report_stat(do_report: bool,
                             s: &str,
                             t: &mut Stat) {
        let e = Gc::stderr_fd();
        if do_report {
            e.write_str("    ");
            e.write_str(s);
            e.write_str(": ");
            e.write_uint(t.curr as uint);
        }
        t.flush();
        if do_report {
            e.write_str(" curr, ");
            e.write_uint(t.total as uint);
            e.write_str(" total\n");
            e.write_str("        ");
            e.write_str("hist: ");
            t.draw_hist(e);
            e.write_str("\n");
        }
    }

    fn flush_and_report_stats(&mut self, phase: &str) {
        if self.report_gc_stats {
            let e = Gc::stderr_fd();
            e.write_str("\n--- ");
            e.write_str(phase);
            e.write_str(" stats ---\n");
            Gc::write_str_uint("    n_collections",
                               self.n_collections);
        }
        Gc::flush_and_report_stat(self.report_gc_stats,
                                  "n_ns_marking",
                                  &mut self.n_ns_marking);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_ns_sweeping",
                        &mut self.n_ns_sweeping);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_ns_total",
                        &mut self.n_ns_total);

        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_mallocs",
                        &mut self.n_mallocs);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_reallocs",
                        &mut self.n_reallocs);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_explicit_frees",
                        &mut self.n_explicit_frees);

        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_boxes_marked_from_tls",
                        &mut self.n_boxes_marked_from_tls);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_bytes_marked_from_tls",
                        &mut self.n_bytes_marked_from_tls);

        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_boxes_marked_from_stack",
                        &mut self.n_boxes_marked_from_stack);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_bytes_marked_from_stack",
                        &mut self.n_bytes_marked_from_stack);

        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_boxes_swept",
                        &mut self.n_boxes_swept);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_bytes_swept",
                        &mut self.n_bytes_swept);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_boxes_annihilated",
                        &mut self.n_boxes_annihilated);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_bytes_annihilated",
                        &mut self.n_bytes_annihilated);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_boxes_skipped_uniq",
                        &mut self.n_boxes_skipped_uniq);
        Gc::flush_and_report_stat(self.report_gc_stats,
                        "n_boxes_skipped_free",
                        &mut self.n_boxes_skipped_free);
    }

    fn note_alloc(&mut self, ptr: uint, sz: uint, align: uint) {

        self.n_mallocs.curr += 1;
        assert!(self.phase == GcIdle || self.phase == GcStarting);
        let h = HeapRecord {
            size: global_heap::get_box_size(sz, align),
            is_marked: false
        };
        self.debug_str_range("gc::note_malloc", ptr, h.size);
        assert!(self.heap.insert(ptr, h));

        if ! self.actually_gc {
            return;
        }
        self.alloc_count += 1;

        // We allocate when the number of heap objects exceeds a threshold
        // (which we raise if nothing is freed). We _also_ GC every N
        // allocs of "churn", even if the heap has not expanded, on the
        // premise that churn might cause existing retained memory to
        // become garbage. N is set to max(100000, 25%-of-the-heap-count).

        if (self.phase == GcIdle && self.gc_zeal) ||
            self.heap.len() > self.threshold ||
            (self.alloc_count > 10000000 &&
             self.alloc_count > self.heap.len()) {
            self.debug_str("commencing gc at threshold: ");
            self.debug_uint(self.threshold);
            self.debug_str("\n");
            self.alloc_count = 0;
            let prev = self.heap.len();
            unsafe {
                self.gc();
            }
            self.debug_str("gc complete, heap count: ");
            self.debug_uint(self.heap.len());
            self.debug_str(" (freed ");
            self.debug_uint(prev - self.heap.len());
            self.debug_str(" boxes)\n");
            if self.heap.len() * 4 > self.threshold {
                self.debug_str("gc did not recover enough, \
                                raising threshold to: ");
                self.debug_uint(self.threshold * 4);
                self.debug_str("\n");
                self.threshold *= 4;
            }
            if self.debug_gc {
                let x = self.threshold;
                self.threshold *= 100;
                self.actually_gc = false;
                self.debug_gc = false;
                for self.tls_marked.each |p| {
                    Gc::write_str_hex("TLS-marked box", *p);
                    Gc::stderr_fd().write_str("\n---\n");
                    unsafe { Gc::debug_opaque_box(*p); }
                    Gc::stderr_fd().write_str("\n---\n");
                }
                self.tls_marked.clear();
                self.debug_gc = true;
                self.actually_gc = true;
                self.threshold = x;
            }
            unsafe {
                if ! self.precious_freed.is_empty() {
                    let x = self.threshold;
                    self.threshold *= 100;
                    self.actually_gc = false;
                    for self.precious_freed.each |p| {
                        Gc::write_str_hex("freed precious address", *p);
                        Gc::stderr_fd().write_str("\n---\n");
                        Gc::debug_opaque_box(*p);
                        Gc::stderr_fd().write_str("\n---\n");
                        self.precious.remove(p);
                        fail!();
                    }
                    self.actually_gc = true;
                    self.precious_freed.clear();
                    self.threshold = x;
                }
            }
        }

    }

    fn note_free(&mut self, ptr: uint) {
        self.debug_str_hex("gc::note_free", ptr);
        self.n_explicit_frees.curr += 1;
        if self.phase != GcIdle {
            assert!(self.free_buffer.insert(ptr));
        } else {
            assert!(self.heap.remove(&ptr));
            unsafe {
                if self.precious.contains(&ptr) {
                    Gc::write_str_hex("explicitly freed precious address", ptr);
                    Gc::stderr_fd().write_str("\n---\n");
                    Gc::debug_opaque_box(ptr);
                    Gc::stderr_fd().write_str("\n---\n");
                    self.precious.remove(&ptr);
                    fail!();
                }
            }
            if ! self.actually_gc {
                return;
            }
            if self.gc_zeal {
                unsafe {
                    self.gc();
                }
            }
            if self.heap.len() < (self.threshold / 8) {
                self.debug_str("lowering gc threshold to: ");
                self.debug_uint(self.threshold / 4);
                self.debug_str("\n");
                self.threshold /= 4;
            }
        }
    }

    fn note_realloc(&mut self, from: uint, to: uint, sz: uint) {

        self.n_reallocs.curr += 1;
        assert!(self.phase == GcIdle);

        if self.debug_gc {
            let e = Gc::stderr_fd();
            e.write_str("gc::note_realloc: ");
            e.write_hex_uint(from);
            e.write_str(" -> [");
            e.write_hex_uint(to);
            e.write_str(", ");
            e.write_hex_uint(to + sz);
            e.write_str(") = ");
            e.write_uint(sz);
            e.write_str(" bytes\n");
        }
        assert!(self.heap.remove(&from));
        let h = HeapRecord {
            size: sz,
            is_marked: false
        };
        assert!(self.heap.insert(to, h));
        if self.precious.contains(&from) {
            self.precious.remove(&from);
            self.precious.insert(to);
        }
        if self.actually_gc && self.gc_zeal {
            unsafe {
                self.gc();
            }
        }
    }

    fn check_consistency(&mut self, phase: &str) {
        let mut n_reported_inconsistencies = 0;
        let thresh = 1000;
        let mut tmp = TrieMap::new();
        for self.each_live_alloc |box| {
            let box = box as uint;
            if ! self.heap.contains_key(&box) {
                n_reported_inconsistencies += 1;
                let e = Gc::stderr_fd();
                if n_reported_inconsistencies < thresh {
                    e.write_str(phase);
                    Gc::write_str_hex(" inconsistency: gc heap \
                                       missing live alloc ptr",
                                      box);
                } else if n_reported_inconsistencies == thresh {
                    e.write_str(phase);
                    e.write_str(" too many inconsistencies\n");
                }
            }
            assert!(tmp.insert(box, ()));
        }
        for self.heap.each_key |box| {
            if ! tmp.contains_key(box) {
                n_reported_inconsistencies += 1;
                let e = Gc::stderr_fd();
                if n_reported_inconsistencies < 1000 {
                    e.write_str(phase);
                    Gc::write_str_hex(" inconsistency: live allocs \
                                       missing gc heap ptr",
                                      *box);
                } else if n_reported_inconsistencies == 1000 {
                    e.write_str(phase);
                    e.write_str(" >1000 inconsistencies\n");
                }
            }
        }
        if self.heap.len() != tmp.len() {
            Gc::stderr_fd().write_str(phase);
            Gc::write_str_uint(" inconsistency: num gc heap ptr",
                               self.heap.len());
            Gc::stderr_fd().write_str(phase);
            Gc::write_str_uint(" inconsistency: num live alloc ptrs",
                               tmp.len());
        }
    }

    unsafe fn each_live_alloc(&self, f: &fn(box: *mut BoxRepr) -> bool) {
        let box = (*self.task).boxed_region.live_allocs;
        let mut box: *mut BoxRepr = transmute(copy box);
        while box != mut_null() {
            let next = transmute(copy (*box).header.next);
            if ! f(box) {
                break
            }
            box = next
        }
    }

    unsafe fn drop_boxes(actually_drop: bool,
                                debug_flag: bool,
                                boxes_dropped: &mut Stat,
                                bytes_dropped: &mut Stat,
                                boxes_skipped_uniq: &mut Stat,
                                boxes_skipped_free: &mut Stat,
                                free_buffer: &mut TrieSet,
                                precious: &mut TrieSet,
                                precious_freed: &mut TrieSet,
                                each: &fn(&fn(*mut BoxRepr, uint) -> bool)) {

        for each |boxp, _size| {
            if free_buffer.contains(&(boxp as uint)) {
                loop;
            }
            let box: &mut BoxRepr = transmute(boxp);
            if box.header.ref_count == RC_MANAGED_UNIQUE {
                loop;
            }
            if debug_flag {
                Gc::write_str_hex("(drop boxes) pass #1: setting immortal",
                                  boxp as uint);
            }
            if precious.contains(&(boxp as uint)) {
                if debug_flag {
                    Gc::write_str_hex("GC freeing precious pointer",
                                      boxp as uint);
                }
                precious_freed.insert(boxp as uint);
            }
            if actually_drop {
                box.header.ref_count = RC_IMMORTAL;
            }
        }

        for each |boxp, _size| {
            if free_buffer.contains(&(boxp as uint)) {
                loop;
            }
            let box: &BoxRepr = transmute(boxp);
            if box.header.ref_count == RC_MANAGED_UNIQUE {
                loop;
            }
            let tydesc: *TypeDesc = transmute(box.header.type_desc);
            let drop_glue: DropGlue = transmute(((*tydesc).drop_glue, 0));
            if debug_flag {
                Gc::write_str_hex("(drop boxes) pass #2: running drop glue",
                                  boxp as uint);
            }
            if precious.contains(&(boxp as uint)) {
                loop;
            }
            if actually_drop && precious_freed.is_empty() {
                drop_glue(to_unsafe_ptr(&tydesc),
                          transmute(&(*box).data));
            }
        }

        for each |boxp, size| {
            if free_buffer.contains(&(boxp as uint)) {
                boxes_skipped_free.curr += 1;
                loop;
            }
            let box: &BoxRepr = transmute(boxp);
            if box.header.ref_count == RC_MANAGED_UNIQUE {
                boxes_skipped_uniq.curr += 1;
                loop;
            }
            if debug_flag {
                Gc::write_str_hex("(drop boxes) pass #3: freeing",
                                  boxp as uint);
            }
            if precious.contains(&(boxp as uint)) {
                loop;
            }
            if actually_drop && precious_freed.is_empty() {
                boxes_dropped.curr += 1;
                bytes_dropped.curr += size as i64;
                rust_upcall_free(transmute(box));
                assert!(free_buffer.insert(boxp as uint));
            }
        }
    }

    fn count_marking(&mut self, size: uint) {
        if self.phase == GcMarkingStack {
            self.n_boxes_marked_from_stack.curr += 1;
            self.n_bytes_marked_from_stack.curr += size as i64;
        } else {
            self.n_boxes_marked_from_tls.curr += 1;
            self.n_bytes_marked_from_tls.curr += size as i64;
        }
    }

    #[inline(always)]
    unsafe fn mark_trie(addr: &mut uint,
                        sz: &mut uint,
                        already_marked: &mut bool,
                        t: &mut TrieMap<HeapRecord>) -> bool {
        let mut hit = false;
        do t.mutate_prev(*addr) |obj, record| {
            if *addr < obj + record.size {
                *already_marked = record.is_marked;
                *addr = obj;
                *sz = record.size;
                if !record.is_marked {
                    *addr = obj;
                    *sz = record.size;
                    record.is_marked = true;
                    hit = true
                }
            }
        }
        return hit;
    }

    #[inline(always)]
    unsafe fn mark_one(&mut self, mut addr: uint) {
        if addr < self.lowest || addr > self.highest {
            return;
        }
        let mut sz = 0;
        let mut already_marked = false;
        if !Gc::mark_trie(&mut addr,
                          &mut sz,
                          &mut already_marked,
                          &mut self.heap) {
            if already_marked {
                if self.phase == GcMarkingStack {
                    self.debug_str_range("  already marked via stack",
                                         addr, sz);
                } else {
                    self.debug_str_range("  already marked via TLS",
                                         addr, sz);
                }
            }
            return;
        }
        if self.debug_gc {
            if self.phase == GcMarkingStack {
                self.debug_str_range("  marked via stack",
                                     addr, sz);
            } else {
                self.debug_str_range("  marked via TLS",
                                     addr, sz);
                self.tls_marked.insert(addr);
            }
        }
        let adj = 32; //size_of::<BoxHeaderRepr>();
        self.marking_stack.push((addr + adj, sz - adj));
    }

    unsafe fn queue_stack(&mut self) {

        // We want to avoid marking frames below us, so we
        // take a base address from the current frame and
        // pass it into the stack-marker.
        let base = 0;
        let base : uint = transmute(&base);
        let mut segment = (*self.task).stack_segment;
        while segment != null() {
            let mut ptr: uint = transmute(&(*segment).data);
            let mut end: uint = transmute(copy (*segment).end);

            if ptr <= base && base < end {
                self.debug_str_hex("limiting stack-marking to base",
                                   base);
                ptr = base;
            }

            self.debug_str_range("queueing stack segment",
                                 ptr, end-ptr);
            self.marking_stack.push((ptr, end-ptr));
            segment = (*segment).prev;
        }
    }

    unsafe fn mark_queued(&mut self) {
        // This is the very hot marking loop. Do
        // everything possible to make it fast.
        let lowest = self.lowest;
        let highest = self.highest;
        // let heapsz = highest - lowest;
        let usz = size_of::<uint>();

        while (self.marking_stack.len() != 0) {
            let (ptr, sz) = self.marking_stack.pop();
            // Suppose we have a 40 byte @@ value.
            // We pushed (ptr,8) on the stack since
            // box hdr is 32 bytes. usz is 8 so we want
            // 1 iteration.
            let mut curr = ptr;
            let end = ptr + sz - usz;
            self.count_marking(sz);
            self.debug_str_range("marking queued range",
                                 ptr, sz);
            while curr <= end {
                let addr = *(curr as *uint);
                curr += 1;
                if addr < lowest || addr > highest {
                    loop;
                }
                /*
                if (addr - lowest) > heapsz {
                    // Implausible address.
                    loop;
                }
                */
                self.mark_one(addr);
            }
            self.debug_str_range("finished scanning range",
                                 ptr, sz);
        }
    }

    unsafe fn mark_extents(&mut self) {
        vec::reserve(&mut self.marking_stack, self.heap.len());

        self.lowest = match self.heap.next(0) {
            None => 0,
            Some((k, _)) => k
        };
        self.highest = match self.heap.prev(uint::max_value) {
            None => uint::max_value,
            Some((k, r)) => k + r.size
        };

        self.debug_str_hex("lowest heap ptr", self.lowest);
        self.debug_str_hex("highest heap ptr", self.highest);

        self.phase = GcMarkingTls;
        self.debug_str("marking TLS values\n");
        do each_retained_ptr(transmute(self.task)) |p| {
            self.debug_str_hex("marking TLS value",
                               transmute(p));
            self.mark_one(transmute(p));
        }
        self.mark_queued();

        self.debug_str("marking stack\n");
        self.phase = GcMarkingStack;
        self.queue_stack();
        self.mark_queued();

    }

    unsafe fn sweep(&mut self) {
        self.phase = GcSweeping;

        // recycle the marking stack here as a scratch
        // space for queueing up the pointers to be
        // freed.
        assert!(self.marking_stack.len() == 0);
        for self.heap.mutate_values |&addr, record| {
            if record.is_marked {
                record.is_marked = false
            } else {
                self.marking_stack.push((addr,
                                         record.size));
            }
        }

        do Gc::drop_boxes(self.actually_gc,
                          self.debug_gc,
                          &mut self.n_boxes_swept,
                          &mut self.n_bytes_swept,
                          &mut self.n_boxes_skipped_uniq,
                          &mut self.n_boxes_skipped_free,
                          &mut self.free_buffer,
                          &mut self.precious,
                          &mut self.precious_freed) |step| {
            for self.marking_stack.each |&(ptr,size)| {
                step(transmute(ptr), size);
            }
        }

        self.marking_stack.clear();

        // Clean out the free buffer
        for self.free_buffer.each |ptr| {
            assert!(self.heap.remove(ptr));
            self.precious.remove(ptr);
            ()
        }

        self.free_buffer.clear();
    }

    unsafe fn set_precious(&mut self, addr: uint) {
        if self.debug_gc {
            self.debug_gc = false;
            self.report_gc_stats = false;
            let x = self.actually_gc;
            self.actually_gc = false;

            Gc::write_str_hex("registered precious pointer",
                              addr);
            Gc::stderr_fd().write_str("\n---\n");
            Gc::debug_opaque_box(addr);
            Gc::stderr_fd().write_str("\n---\n");

            self.actually_gc = x;
            self.report_gc_stats = true;
            self.debug_gc = true;
        }
        self.precious.insert(addr);
    }

    unsafe fn gc(&mut self) {

        let mut start = 0;
        let mut end = 0;
        let mut mark_start = 0;
        let mut mark_end = 0;
        let mut sweep_start = 0;
        let mut sweep_end = 0;

        self.phase = GcStarting;

        precise_time_ns(&mut start);

        // NB: need this here before we lock down the GC, to make sure
        // TLS is initialized; easiest approach. It allocates a @Dvec.
        do each_retained_ptr(transmute(self.task)) |_| { }

        self.debug_str("gc starting\n");
        if self.check_gc_consistency {
            self.check_consistency("pre-gc");
        }

        precise_time_ns(&mut mark_start);
        self.mark_extents();
        precise_time_ns(&mut mark_end);

        precise_time_ns(&mut sweep_start);
        self.sweep();
        precise_time_ns(&mut sweep_end);

        if self.check_gc_consistency {
                self.check_consistency("post-gc");
        }
        self.debug_str("gc finished\n");

        self.phase = GcIdle;

        self.n_collections += 1;
        precise_time_ns(&mut end);

        self.n_ns_marking.curr += (mark_end - mark_start) as i64;
        self.n_ns_sweeping.curr += (sweep_end - sweep_start) as i64;
        self.n_ns_total.curr += (end - start) as i64;

        self.flush_and_report_stats("gc");
    }

    unsafe fn annihilate(&mut self) {
        self.debug_str("annihilation starting\n");
        self.check_consistency("pre-annihilate");
        self.phase = GcAnnihilating;

        // Quick hack: we need to free this list upon task exit, and this
        // is a convenient place to do it.
        clear_task_borrow_list();

        self.precious.clear();
        do Gc::drop_boxes(true,
                          self.debug_gc,
                          &mut self.n_boxes_annihilated,
                          &mut self.n_bytes_annihilated,
                          &mut self.n_boxes_skipped_uniq,
                          &mut self.n_boxes_skipped_free,
                          &mut self.free_buffer,
                          &mut self.precious,
                          &mut self.precious_freed) |step| {
            for self.heap.mutate_values |&ptr, record| {
                step(transmute(ptr), record.size);
            }
        }
        for self.free_buffer.each |ptr| {
            assert!(self.heap.remove(ptr));
            ()
        }
        self.phase = GcIdle;
        self.check_consistency("post-annihilate");
        self.debug_str("annihilation finished\n");
        self.flush_and_report_stats("annihilation");
    }
}

pub struct Stat {
    curr: i64,
    total: i64,
    hist: [i64, ..64],
}

static hist_bars : [char, ..8] = ['▁','▂','▃','▄','▅','▆','▇','█'];
static empty : char = '⋯';

pub impl Stat {
    fn flush(&mut self) {
        self.total += self.curr;
        unsafe {
            self.hist[64 - ctlz64(self.curr)] += 1;
        }
        self.curr = 0;
    }

    fn draw_hist<W:WriterUtil>(&self, w: W) {
        w.write_char('[');
        for vec::each(self.hist) |elt| {
            if *elt == 0 {
                w.write_char(empty);
            } else {
                let magnitude = 64 - unsafe { ctlz64(*elt) };
                let magnitude = i64::min(magnitude, 8);
                w.write_char(hist_bars[magnitude - 1]);
            }
        }
        w.write_char(']');
    }

    fn new() -> Stat {
        Stat {
            curr: 0,
            total: 0,
            hist: [0, ..64]
        }
    }
}


/// Releases all managed memory (i.e. @ boxes) in the current task.
#[cfg(not(test))]
#[lang="annihilate"]
pub unsafe fn annihilate() {
    let gc : &mut Gc = Gc::get_task_gc();
    gc.annihilate();
    let task: &Task = transmute(rust_get_task());
    let _dropme: ~Gc = transmute(task.gc);
}

fn gc_phase_2() {
    unsafe {
        Gc::get_task_gc().gc();
    }
}


/// Releases all unreachable managed memory in the current task.
#[cfg(stage0)]
pub fn gc() {

    unsafe {
        // Awkward but necessary: registers must be 16-byte
        // aligned on x64 due to dumping xmm state using
        // movapd. So we don't use the Registers structure.
        let regs = [0u64, ..80];
        let r : uint = transmute(&regs);
        let r = (r + 16) & (!15);

        // Dump registers to stack and jump to gc_phase_2
        let f : *c_void = transmute(gc_phase_2);
        dump_registers(transmute(r), f);
    }
}

/// Releases all unreachable managed memory in the current task.
#[cfg(stage1)]
#[cfg(stage2)]
#[cfg(stage3)]
pub fn gc() {

    unsafe {
        // Awkward but necessary: registers must be 16-byte
        // aligned on x64 due to dumping xmm state using
        // movapd. So we don't use the Registers structure.
        let regs = [0u64, ..80];
        let r : uint = transmute(&regs);
        let r = (r + 16) & (!15);

        // Dump registers to stack and jump to gc_phase_2
        let f : *c_void = transmute(gc_phase_2);
        dump_registers(transmute(r), f);
    }
}

/**
* Register a box-address as "precious", meaning the GC will report
* an error if it tries to free the address before annihilation
* (i.e. before TLS-teardown).
*/
#[allow(managed_heap_memory)]
pub fn set_precious<T>(t: @T) {
    unsafe {
        Gc::get_task_gc().set_precious(transmute(t));
    }
}

#[link_name = "rustrt"]
extern {
    fn precise_time_ns(ns: &mut u64);

    #[rust_stack]
    // FIXME (#4386): Unable to make following method private.
    fn rust_get_task() -> *c_void;

    #[rust_stack]
    fn rust_upcall_free(ptr: *c_char);

    #[rust_stack]
    fn dump_registers(out_regs: *mut Registers, next_fn: *c_void);
}

