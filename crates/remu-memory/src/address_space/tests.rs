// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

const SPACE_SIZE: u64 = 16 * GUEST_PAGE_SIZE;

fn page(base: GuestAddress, index: u64) -> GuestAddress {
    GuestAddress::new(base.get() + index * GUEST_PAGE_SIZE)
}

#[test]
fn creates_aligned_guarded_address_space_and_queries_free_gap() {
    let memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    assert_eq!(memory.base().get() % GUEST_PAGE_SIZE, 0);
    assert_eq!(memory.size(), SPACE_SIZE);
    assert_eq!(
        memory.query(memory.base()).unwrap(),
        GuestVmaInfo {
            start: memory.base(),
            size: SPACE_SIZE,
            state: GuestVmaState::Free,
            protection: GuestProtection::NONE,
            backing_id: None,
            backing_offset: 0,
            metadata: 0,
        }
    );
}

#[test]
fn reservations_and_hints_have_deterministic_first_fit_order() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let base = memory.base();
    let middle = memory
        .reserve(
            GuestMapPlacement::Fixed(page(base, 4)),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    assert_eq!(middle, page(base, 4));
    let after = memory
        .reserve(
            GuestMapPlacement::Hint(page(base, 4)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    assert_eq!(after, page(base, 6));
    let first = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    assert_eq!(first, base);
}

#[test]
fn fixed_mapping_commits_a_reservation_but_rejects_live_mapping_overlap() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let target = page(memory.base(), 2);
    memory
        .reserve(
            GuestMapPlacement::Fixed(target),
            3 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    memory
        .map_anonymous(
            GuestMapPlacement::Fixed(page(target, 1)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();
    assert_eq!(memory.vmas().len(), 3);
    let error = memory
        .map_anonymous(
            GuestMapPlacement::Fixed(page(target, 1)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AddressSpaceError::Overlap {
            state: GuestVmaState::Committed,
            ..
        }
    ));
}

#[test]
fn protect_splits_and_then_merges_anonymous_vmas() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let base = memory
        .map_anonymous(
            GuestMapPlacement::Anywhere,
            3 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();
    memory
        .protect(page(base, 1), GUEST_PAGE_SIZE, GuestProtection::READ)
        .unwrap();
    let vmas = memory.vmas();
    assert_eq!(vmas.len(), 3);
    assert_eq!(vmas[1].protection, GuestProtection::READ);
    memory
        .protect(page(base, 1), GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();
    assert_eq!(memory.vmas().len(), 1);
}

#[test]
fn protection_commits_only_the_selected_reserved_pages() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let reserved = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            3 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();

    let committed = page(reserved, 1);
    memory
        .protect(committed, GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();
    memory.write(committed, b"committed").unwrap();

    assert_eq!(
        memory.query(reserved).unwrap().state,
        GuestVmaState::Reserved
    );
    assert_eq!(
        memory.query(committed).unwrap().state,
        GuestVmaState::Committed
    );
    assert_eq!(
        memory.query(page(reserved, 2)).unwrap().state,
        GuestVmaState::Reserved
    );
    assert!(matches!(
        memory.map_anonymous(
            GuestMapPlacement::Fixed(committed),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        ),
        Err(AddressSpaceError::Overlap {
            state: GuestVmaState::Committed,
            ..
        })
    ));
}

#[test]
fn opaque_metadata_follows_split_and_merge_range_surgery() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let base = memory
        .map_anonymous(
            GuestMapPlacement::Anywhere,
            3 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();
    memory
        .set_metadata(page(base, 1), GUEST_PAGE_SIZE, 42)
        .unwrap();
    let split = memory.vmas();
    assert_eq!(split.len(), 3);
    assert_eq!(split[1].metadata, 42);
    memory.set_metadata(base, 3 * GUEST_PAGE_SIZE, 7).unwrap();
    let merged = memory.vmas();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].metadata, 7);
}

#[test]
fn unmap_splits_vma_and_freed_hole_is_reused() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let base = memory
        .map_anonymous(
            GuestMapPlacement::Anywhere,
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();
    memory.unmap(page(base, 1), 2 * GUEST_PAGE_SIZE).unwrap();
    assert_eq!(memory.vmas().len(), 2);
    assert_eq!(
        memory.query(page(base, 1)).unwrap().state,
        GuestVmaState::Free
    );
    let reused = memory
        .map_anonymous(
            GuestMapPlacement::Hint(page(base, 1)),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();
    assert_eq!(reused, page(base, 1));
    assert_eq!(memory.vmas().len(), 1);
}

#[test]
fn shared_backing_aliases_are_coherent() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let first = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let second = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let backing_id = backing.id();
    drop(backing);
    memory.write(page(first, 1), b"alias").unwrap();
    let mut bytes = [0_u8; 5];
    memory.read(page(second, 1), &mut bytes).unwrap();
    assert_eq!(&bytes, b"alias");
    assert_eq!(memory.query(second).unwrap().backing_id, Some(backing_id));
    assert_eq!(
        guest_shared_mapping_identity(first.get(), 2 * GUEST_PAGE_SIZE),
        Some(GuestSharedMappingIdentity {
            backing_id,
            backing_offset: 0,
        })
    );
    assert_eq!(
        guest_shared_mapping_identity(page(second, 1).get(), GUEST_PAGE_SIZE),
        Some(GuestSharedMappingIdentity {
            backing_id,
            backing_offset: GUEST_PAGE_SIZE,
        })
    );
}

#[test]
fn private_backing_write_is_coherent_without_falsely_dirtying_cpu_watches() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let first = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let second = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    // SAFETY: both mappings remain live until the watches are dropped at the end of the test.
    let first_watch =
        unsafe { CpuWriteWatch::new(first.get(), GUEST_PAGE_SIZE).expect("first watch") };
    // SAFETY: same backing and lifetime guarantee as the first alias.
    let second_watch =
        unsafe { CpuWriteWatch::new(second.get(), GUEST_PAGE_SIZE).expect("second watch") };

    assert!(write_guest_shared_backing(first.get(), b"gpu"));
    let mut coherent = [0_u8; 3];
    memory.read(second, &mut coherent).unwrap();
    assert_eq!(&coherent, b"gpu");
    assert_eq!(first_watch.generation(), 0);
    assert_eq!(second_watch.generation(), 0);
    assert!(!first_watch.is_dirty());
    assert!(!second_watch.is_dirty());

    // Fault ownership is virtual-address-specific. A texture watcher covers every physical
    // alias and therefore observes this store through `first_watch`; the untouched alias does
    // not need a duplicate dirty transition.
    memory.write(first, b"cpu").unwrap();
    assert!(first_watch.is_dirty());
    assert!(!second_watch.is_dirty());
    assert_eq!(first_watch.generation(), 1);
    assert_eq!(second_watch.generation(), 0);

    assert!(write_guest_shared_backing(first.get(), b"gpu"));
    assert!(first_watch.is_dirty());
    assert_eq!(first_watch.generation(), 1);
}

#[test]
fn identity_guarded_backing_write_rejects_reused_guest_address() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let first_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &first_backing,
            0,
        )
        .unwrap();
    let first_identity = guest_shared_mapping_identity(mapped.get(), GUEST_PAGE_SIZE)
        .expect("first mapping identity");
    assert!(write_guest_shared_backing_if_identity(
        mapped.get(),
        first_identity,
        b"first"
    ));

    memory.unmap(mapped, GUEST_PAGE_SIZE).unwrap();
    let replacement_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(mapped),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement_backing,
            0,
        )
        .unwrap();

    assert!(!write_guest_shared_backing_if_identity(
        mapped.get(),
        first_identity,
        b"stale"
    ));
    let mut replacement = [0_u8; 5];
    memory.read(mapped, &mut replacement).unwrap();
    assert_eq!(replacement, [0; 5]);
}

#[test]
fn identity_targeted_backing_write_survives_same_heap_va_recycle() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let first = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let first_identity = guest_shared_backing_identity(first.get(), GUEST_PAGE_SIZE)
        .expect("first mapping identity");
    let observer = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();

    memory.unmap(first, GUEST_PAGE_SIZE).unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(first),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    let recycled_identity = guest_shared_backing_identity(first.get(), GUEST_PAGE_SIZE)
        .expect("recycled mapping identity");
    assert_ne!(first_identity, recycled_identity);

    assert!(write_guest_shared_backing_at_identity(
        first_identity,
        b"glyph"
    ));
    let mut observed = [0_u8; 5];
    memory.read(observer, &mut observed).unwrap();
    assert_eq!(&observed, b"glyph");
    let mut recycled = [0_u8; 5];
    memory.read(first, &mut recycled).unwrap();
    assert_eq!(recycled, [0; 5]);
}

#[test]
fn identity_targeted_backing_write_is_refused_for_a_virtually_keyed_backing() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let committed = memory
        .map_committed(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();
    let identity = guest_shared_backing_identity(committed.get(), GUEST_PAGE_SIZE)
        .expect("committed mapping identity");

    assert!(!write_guest_shared_backing_at_identity(identity, b"glyph"));
    let mut observed = [0_u8; 5];
    memory.read(committed, &mut observed).unwrap();
    assert_eq!(observed, [0; 5]);
}

#[test]
fn identity_targeted_backing_write_is_refused_when_live_records_cover_only_part_of_the_range() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let identity = guest_shared_backing_identity(mapped.get(), 2 * GUEST_PAGE_SIZE)
        .expect("two-page mapping identity");

    memory.unmap(page(mapped, 1), GUEST_PAGE_SIZE).unwrap();

    let source = vec![0xab_u8; usize::try_from(GUEST_PAGE_SIZE).unwrap() + 16];
    assert!(!write_guest_shared_backing_at_identity(identity, &source));
    let mut observed = [0_u8; 5];
    memory.read(mapped, &mut observed).unwrap();
    assert_eq!(observed, [0; 5]);
}

#[test]
fn identity_targeted_backing_write_publishes_a_fully_covered_physical_range() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let identity = guest_shared_backing_identity(page(mapped, 1).get(), GUEST_PAGE_SIZE)
        .expect("second page identity");

    assert!(write_guest_shared_backing_at_identity(identity, b"glyph"));
    let mut observed = [0_u8; 5];
    memory.read(page(mapped, 1), &mut observed).unwrap();
    assert_eq!(&observed, b"glyph");
}

#[test]
fn a_lease_publishes_a_clean_run_at_the_live_va() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    memory.write(mapped, &[0x11; 8]).unwrap();
    let lease = GuestRangeLease::acquire(mapped.get(), 8).expect("live range leases");

    assert_eq!(
        lease.publish_run_if_baseline(0, &[0x11; 8], &[0x5a; 8]),
        LeasePublishOutcome::AppliedAtLiveVa
    );
    let mut observed = [0_u8; 8];
    memory.read(mapped, &mut observed).unwrap();
    assert_eq!(observed, [0x5a; 8]);
}

#[test]
fn a_lease_yields_to_a_guest_store_after_capture() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    memory.write(mapped, &[0x11; 8]).unwrap();
    let lease = GuestRangeLease::acquire(mapped.get(), 8).expect("live range leases");

    // The guest recycles the range after the capture.
    memory.write(mapped, &[0x77; 8]).unwrap();

    assert_eq!(
        lease.publish_run_if_baseline(0, &[0x11; 8], &[0x5a; 8]),
        LeasePublishOutcome::GuestModified
    );
    let mut observed = [0_u8; 8];
    memory.read(mapped, &mut observed).unwrap();
    assert_eq!(observed, [0x77; 8], "the guest's newer bytes survive");
}

#[test]
fn shared_64_bit_label_publishes_as_one_release_store() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let identity = guest_shared_mapping_identity(mapped.get(), 8).expect("label identity");
    let label = 0x1122_3344_5566_7788_u64;
    assert!(publish_guest_shared_u64_if_identity(
        mapped.get(),
        identity,
        label
    ));
    assert_eq!(load_guest_shared_u64(mapped.get()), Some(label));
    assert_eq!(load_guest_visible_u64(mapped.get()), label);
}

#[test]
fn word_sized_baseline_guard_uses_compare_exchange() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    store_guest_visible_u64(mapped.get(), 0x1111_1111_1111_1111);
    let identity = guest_shared_mapping_identity(mapped.get(), 8).expect("word identity");
    assert_eq!(
        write_guest_shared_backing_if_identity_and_baseline(
            mapped.get(),
            identity,
            &0x1111_1111_1111_1111_u64.to_le_bytes(),
            &0x5a5a_5a5a_5a5a_5a5a_u64.to_le_bytes(),
        ),
        GuardedBackingWrite::Applied
    );
    store_guest_visible_u64(mapped.get(), 0x7777_7777_7777_7777);
    assert_eq!(
        write_guest_shared_backing_if_identity_and_baseline(
            mapped.get(),
            identity,
            &0x5a5a_5a5a_5a5a_5a5a_u64.to_le_bytes(),
            &0xaaaa_aaaa_aaaa_aaaa_u64.to_le_bytes(),
        ),
        GuardedBackingWrite::GuestModified
    );
    assert_eq!(load_guest_visible_u64(mapped.get()), 0x7777_7777_7777_7777);
}

#[test]
fn a_lease_follows_a_recycled_va_through_the_identity_licence() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let observer = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let lease = GuestRangeLease::acquire(mapped.get(), 8).expect("live range leases");

    // Prospero recycles the producer VA onto a different backing offset mid-flight.
    memory.unmap(mapped, GUEST_PAGE_SIZE).unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(mapped),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .unwrap();

    assert_eq!(
        lease.publish_run_if_baseline(0, &[0; 8], &[0x5a; 8]),
        LeasePublishOutcome::AppliedByIdentity
    );
    let mut observed = [0_u8; 8];
    memory.read(observer, &mut observed).unwrap();
    assert_eq!(
        observed, [0x5a; 8],
        "the original physical pages hold the GPU result"
    );
}

#[test]
fn a_live_va_publication_is_revoked_when_the_va_is_recycled() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let observer = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    memory.write(mapped, &[0x11; 8]).unwrap();
    let lease = GuestRangeLease::acquire(mapped.get(), 8).expect("live range leases");

    memory.unmap(mapped, GUEST_PAGE_SIZE).unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(mapped),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .unwrap();

    assert_eq!(
        lease.publish_run_at_live_va_if_baseline(0, &[0x11; 8], &[0x5a; 8]),
        LeasePublishOutcome::Revoked
    );
    let mut observed = [0_u8; 8];
    memory.read(observer, &mut observed).unwrap();
    assert_eq!(
        observed, [0x11; 8],
        "the recycled physical pages are untouched"
    );
}

#[test]
fn a_lease_yields_to_guest_stores_on_recycled_physical_pages() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let observer = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let lease = GuestRangeLease::acquire(mapped.get(), 8).expect("live range leases");

    memory.unmap(mapped, GUEST_PAGE_SIZE).unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(mapped),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    // The allocator hands the original physical pages out and the guest stores through the
    // surviving alias before the delayed publication lands.
    memory.write(observer, &[0x77; 8]).unwrap();

    assert_eq!(
        lease.publish_run_if_baseline(0, &[0; 8], &[0x5a; 8]),
        LeasePublishOutcome::GuestModified
    );
    let mut observed = [0_u8; 8];
    memory.read(observer, &mut observed).unwrap();
    assert_eq!(observed, [0x77; 8], "recycled heap keeps the guest's bytes");
}

#[test]
fn a_lease_is_revoked_when_the_backing_loses_every_live_mapping() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let lease = GuestRangeLease::acquire(mapped.get(), 8).expect("live range leases");

    memory.unmap(mapped, GUEST_PAGE_SIZE).unwrap();

    assert_eq!(
        lease.publish_run_if_baseline(0, &[0; 8], &[0x5a; 8]),
        LeasePublishOutcome::Revoked
    );
}

#[test]
fn replacing_shared_pages_removes_their_backing_identity() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    assert!(guest_shared_mapping_identity(mapped.get(), GUEST_PAGE_SIZE).is_some());

    memory.unmap(mapped, 2 * GUEST_PAGE_SIZE).unwrap();

    assert_eq!(
        guest_shared_mapping_identity(mapped.get(), GUEST_PAGE_SIZE),
        None
    );
}

#[test]
fn contiguous_shared_slices_merge_but_discontinuous_slices_do_not() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).unwrap();
    let base = memory.base();
    for (virtual_page, physical_page) in [(0, 0), (1, 1), (2, 3)] {
        memory
            .map_shared(
                GuestMapPlacement::Fixed(page(base, virtual_page)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                physical_page * GUEST_PAGE_SIZE,
            )
            .unwrap();
    }
    let vmas = memory.vmas();
    assert_eq!(vmas.len(), 2);
    assert_eq!(vmas[0].size, 2 * GUEST_PAGE_SIZE);
    assert_eq!(vmas[1].backing_offset, 3 * GUEST_PAGE_SIZE);
}

#[test]
fn gpu_backing_range_spans_disjoint_virtual_aliases() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).unwrap();
    let base = memory.base();
    let first = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 0)),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let second = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 8)),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            2 * GUEST_PAGE_SIZE,
        )
        .unwrap();
    let source = (0..4 * usize::try_from(GUEST_PAGE_SIZE).unwrap())
        .map(|index| u8::try_from(index % 251).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(
        guest_shared_mapping_identity(first.get(), 4 * GUEST_PAGE_SIZE),
        None,
        "the CPU virtual interval contains a guard hole"
    );
    assert_eq!(
        guest_shared_backing_segments(first.get(), 4 * GUEST_PAGE_SIZE),
        Some(vec![
            GuestSharedBackingSegment {
                address: first.get(),
                byte_count: 2 * GUEST_PAGE_SIZE,
            },
            GuestSharedBackingSegment {
                address: second.get(),
                byte_count: 2 * GUEST_PAGE_SIZE,
            },
        ])
    );
    assert!(write_guest_shared_backing(first.get(), &source));
    let mut second_half = vec![0; usize::try_from(2 * GUEST_PAGE_SIZE).unwrap()];
    memory.read(second, &mut second_half).unwrap();
    assert_eq!(
        second_half,
        source[usize::try_from(2 * GUEST_PAGE_SIZE).unwrap()..]
    );
    let mut round_trip = vec![0; source.len()];
    assert!(read_guest_shared_backing(first.get(), &mut round_trip));
    assert_eq!(round_trip, source);
}

#[test]
fn gpu_backing_range_rejects_unmapped_physical_holes() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).unwrap();
    let base = memory.base();
    let first = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 0)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 8)),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            2 * GUEST_PAGE_SIZE,
        )
        .unwrap();

    assert_eq!(
        guest_shared_backing_segments(first.get(), 4 * GUEST_PAGE_SIZE),
        None
    );
    assert!(!read_guest_shared_backing(
        first.get(),
        &mut vec![0; usize::try_from(4 * GUEST_PAGE_SIZE).unwrap()]
    ));
}

#[test]
fn repeated_gpu_backing_reads_reuse_a_coherent_mapping() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();

    memory.write(mapped, b"first").unwrap();
    let mut bytes = [0_u8; 5];
    assert!(read_guest_shared_backing(mapped.get(), &mut bytes));
    assert_eq!(&bytes, b"first");
    assert_eq!(backing.read_mapping_cache_stats(), (1, 0, 1));

    assert!(write_guest_shared_backing(mapped.get(), b"later"));
    assert!(read_guest_shared_backing(mapped.get(), &mut bytes));
    assert_eq!(
        &bytes, b"later",
        "the cached MAP_SHARED view stays coherent"
    );
    let interior = GuestAddress::new(mapped.get() + 1);
    let mut interior_bytes = [0_u8; 3];
    assert!(read_guest_shared_backing(
        interior.get(),
        &mut interior_bytes
    ));
    assert_eq!(&interior_bytes, b"ate");
    assert_eq!(
        backing.read_mapping_cache_stats(),
        (1, 2, 1),
        "the exact repeat and contained read must avoid new mmap calls",
    );
}

#[test]
fn gpu_backing_read_mapping_cache_stays_bounded() {
    const PAGE_COUNT: u64 = 257;
    let backing = GuestPhysicalBacking::new(PAGE_COUNT * GUEST_PAGE_SIZE).unwrap();
    let mut byte = [0_u8; 1];
    for page_index in 0..PAGE_COUNT {
        assert!(backing.read_exact_at(page_index * GUEST_PAGE_SIZE, &mut byte));
    }
    assert_eq!(
        backing.read_mapping_cache_stats(),
        (256, 0, PAGE_COUNT),
        "the least-recently-used view is retired at the fixed entry limit",
    );
}

#[test]
fn repeated_backing_writes_reuse_a_cached_writable_mapping() {
    // Every writeback used to mmap and munmap a fresh window; a commit-heavy title publishing
    // per-row texture writebacks paid two syscalls per row. Mirror the read cache so a hot
    // physical window is mapped once and reused.
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    assert!(backing.write_all_at(0, b"first"));
    assert_eq!(
        backing.write_mapping_cache_stats(),
        (1, 0, 1),
        "the first write installs the writable window as a miss"
    );
    assert!(backing.write_all_at(8, b"second"));
    assert_eq!(
        backing.write_mapping_cache_stats(),
        (1, 1, 1),
        "a second write to the same page reuses the cached window without a new mmap"
    );

    // The cached MAP_SHARED window stays coherent with the read path over the same object.
    let mut bytes = [0_u8; 6];
    assert!(backing.read_exact_at(8, &mut bytes));
    assert_eq!(&bytes, b"second");
}

#[test]
fn backing_reads_and_writes_refuse_out_of_range_offsets() {
    // M3 fault injection: the size check is the enforcer that keeps the unsafe mmap/copy in
    // backing.rs from ever seeing an offset outside the object. Every out-of-range request must be
    // rejected before any unsafe code runs, and in-range neighbours must still succeed.
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let mut scratch = [0_u8; 8];

    // Offset at the very end, and one that overflows u64, are rejected without a mapping.
    assert!(!backing.write_all_at(GUEST_PAGE_SIZE, b"x"));
    assert!(!backing.read_exact_at(GUEST_PAGE_SIZE, &mut scratch[..1]));
    assert!(!backing.write_all_at(u64::MAX, b"x"));
    assert!(!backing.read_exact_at(u64::MAX, &mut scratch[..1]));
    // A write that starts in range but runs past the end is rejected whole.
    assert!(!backing.write_all_at(GUEST_PAGE_SIZE - 4, &[0_u8; 8]));
    assert_eq!(
        backing.write_mapping_cache_stats(),
        (0, 0, 0),
        "a rejected write never installs a cached window"
    );

    // The last in-range bytes still round-trip.
    assert!(backing.write_all_at(GUEST_PAGE_SIZE - 8, b"tailend!"));
    assert!(backing.read_exact_at(GUEST_PAGE_SIZE - 8, &mut scratch));
    assert_eq!(&scratch, b"tailend!");
}

#[test]
fn backing_write_mapping_cache_stays_bounded() {
    const PAGE_COUNT: u64 = 257;
    let backing = GuestPhysicalBacking::new(PAGE_COUNT * GUEST_PAGE_SIZE).unwrap();
    for page_index in 0..PAGE_COUNT {
        assert!(backing.write_all_at(page_index * GUEST_PAGE_SIZE, b"x"));
    }
    assert_eq!(
        backing.write_mapping_cache_stats(),
        (256, 0, PAGE_COUNT),
        "the least-recently-used writable window is retired at the fixed entry limit",
    );
}

#[test]
fn gpu_backing_alias_enumeration_tracks_aliases_and_remaps() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let replacement = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let base = memory.base();
    let first = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 0)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let continuation = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 4)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    let alias = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 8)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();

    assert_eq!(
        guest_shared_backing_alias_segments(first.get(), 2 * GUEST_PAGE_SIZE),
        Some(vec![
            GuestSharedBackingSegment {
                address: first.get(),
                byte_count: GUEST_PAGE_SIZE,
            },
            GuestSharedBackingSegment {
                address: continuation.get(),
                byte_count: GUEST_PAGE_SIZE,
            },
            GuestSharedBackingSegment {
                address: alias.get(),
                byte_count: GUEST_PAGE_SIZE,
            },
        ])
    );

    memory.unmap(continuation, GUEST_PAGE_SIZE).unwrap();
    assert_eq!(
        guest_shared_backing_alias_segments(first.get(), 2 * GUEST_PAGE_SIZE),
        None,
        "removing the only continuation invalidates complete backing coverage"
    );
    memory
        .map_shared(
            GuestMapPlacement::Fixed(continuation),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement,
            0,
        )
        .unwrap();
    assert_eq!(
        guest_shared_backing_alias_segments(first.get(), 2 * GUEST_PAGE_SIZE),
        None,
        "a different physical backing cannot repair the resource span"
    );
}

#[test]
fn physical_range_overlap_compares_aliases_without_requiring_matching_virtual_addresses() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let other_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let base = memory.base();
    let first = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 0)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let continuation = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 4)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    let alias = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 8)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let unrelated = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 12)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &other_backing,
            0,
        )
        .unwrap();

    assert!(guest_shared_backing_ranges_overlap(
        first.get(),
        GUEST_PAGE_SIZE,
        alias.get(),
        GUEST_PAGE_SIZE,
    ));
    assert!(!guest_shared_backing_ranges_overlap(
        first.get(),
        GUEST_PAGE_SIZE,
        continuation.get(),
        GUEST_PAGE_SIZE,
    ));
    assert!(guest_shared_backing_ranges_overlap(
        first.get(),
        2 * GUEST_PAGE_SIZE,
        continuation.get(),
        GUEST_PAGE_SIZE,
    ));
    assert!(!guest_shared_backing_ranges_overlap(
        first.get(),
        GUEST_PAGE_SIZE,
        unrelated.get(),
        GUEST_PAGE_SIZE,
    ));
    assert!(!guest_shared_backing_ranges_overlap(
        first.get(),
        0,
        alias.get(),
        GUEST_PAGE_SIZE,
    ));
    let first_identity =
        guest_shared_backing_identity(first.get(), GUEST_PAGE_SIZE).expect("first identity");
    assert_eq!(
        guest_shared_backing_identity(alias.get(), GUEST_PAGE_SIZE),
        Some(first_identity),
    );
    assert_eq!(
        guest_shared_backing_identity(first.get(), 2 * GUEST_PAGE_SIZE),
        Some(first_identity),
        "physical identity accepts a contiguous backing exposed through disjoint virtual maps",
    );
    assert_ne!(
        guest_shared_backing_identity(continuation.get(), GUEST_PAGE_SIZE),
        Some(first_identity),
    );
    let _ = take_guest_shared_identity_statistics();
    assert!(guest_shared_backing_any_range_overlaps_resolved(
        [(alias.get(), GUEST_PAGE_SIZE)],
        first.get(),
        GUEST_PAGE_SIZE,
        Some(first_identity),
    ));
    assert!(!guest_shared_backing_any_range_overlaps_resolved(
        [
            (continuation.get(), GUEST_PAGE_SIZE),
            (unrelated.get(), GUEST_PAGE_SIZE),
        ],
        first.get(),
        GUEST_PAGE_SIZE,
        Some(first_identity),
    ));
    assert_eq!(
        take_guest_shared_identity_statistics(),
        (0, 0),
        "the batched fallback must reuse the fixed identity instead of resolving it again",
    );
}

#[test]
fn no_coalesce_shared_maps_retain_independent_query_boundaries() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let base = memory.base();
    for page_index in 0..2 {
        memory
            .map_shared_with_coalescing(
                GuestMapPlacement::Fixed(page(base, page_index)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                GuestSharedMapOptions {
                    backing_offset: page_index * GUEST_PAGE_SIZE,
                    coalescing: GuestMapCoalescing::Disallowed,
                },
            )
            .unwrap();
    }
    let vmas = memory.vmas();
    assert_eq!(vmas.len(), 2);
    assert!(vmas.iter().all(|vma| vma.size == GUEST_PAGE_SIZE));
    assert_eq!(memory.query(base).unwrap().start, base);
    assert_eq!(memory.query(page(base, 1)).unwrap().start, page(base, 1));
    memory.unmap(base, 2 * GUEST_PAGE_SIZE).unwrap();
    assert!(memory.vmas().is_empty());
}

#[test]
fn read_write_helpers_enforce_permissions_and_mapping_coverage() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let base = memory
        .map_anonymous(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ,
        )
        .unwrap();
    assert!(matches!(
        memory.write(base, b"denied").unwrap_err(),
        AddressSpaceError::PermissionDenied { .. }
    ));
    assert!(matches!(
        memory.read(page(base, 1), &mut [0_u8; 1]).unwrap_err(),
        AddressSpaceError::NotMapped { .. }
    ));
}

#[test]
fn rejects_alignment_bounds_backing_and_overflow_errors() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    assert!(matches!(
        memory.reserve(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE - 1,
            GUEST_PAGE_SIZE
        ),
        Err(AddressSpaceError::Unaligned { .. })
    ));
    assert!(matches!(
        memory.reserve(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE / 2
        ),
        Err(AddressSpaceError::InvalidAlignment { .. })
    ));
    assert!(matches!(
        memory.map_anonymous(
            GuestMapPlacement::Fixed(GuestAddress::new(!(GUEST_PAGE_SIZE - 1))),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        ),
        Err(AddressSpaceError::AddressOverflow { .. })
    ));
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    assert!(matches!(
        memory.map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        ),
        Err(AddressSpaceError::BackingOutOfBounds { .. })
    ));
}

#[test]
fn unmap_is_atomic_on_unmapped_gap_validation_failure() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let base = memory
        .map_anonymous(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();
    let before = memory.vmas();
    assert!(matches!(
        memory.unmap(base, 2 * GUEST_PAGE_SIZE),
        Err(AddressSpaceError::NotMapped { .. })
    ));
    assert_eq!(memory.vmas(), before);
    memory.write(base, b"still mapped").unwrap();
}

#[test]
fn protect_committed_reserved_pages_have_resolvable_shared_backing() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let reserved = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    memory
        .protect(reserved, 4 * GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();
    memory.write(page(reserved, 1), b"texels").unwrap();

    // The committed range is one live coherent alias of its own physical backing, exactly like
    // a direct-memory mapping: the GPU residency layer must resolve it end to end.
    let interior = page(reserved, 1).get();
    let identity = guest_shared_mapping_identity(interior, 2 * GUEST_PAGE_SIZE)
        .expect("committed pages resolve to a physical backing identity");
    let retained = guest_shared_backing_range(interior, 2 * GUEST_PAGE_SIZE)
        .expect("committed pages resolve to a retained backing view");
    assert!(
        retained.is_virtually_keyed(),
        "committed-memory offsets cannot be retained as durable physical allocation identity"
    );
    assert_eq!(
        guest_shared_backing_segments(interior, 2 * GUEST_PAGE_SIZE),
        Some(vec![GuestSharedBackingSegment {
            address: interior,
            byte_count: 2 * GUEST_PAGE_SIZE,
        }])
    );
    assert_eq!(
        guest_shared_backing_alias_segments(interior, 2 * GUEST_PAGE_SIZE),
        Some(vec![GuestSharedBackingSegment {
            address: interior,
            byte_count: 2 * GUEST_PAGE_SIZE,
        }])
    );
    let mut bytes = [0_u8; 6];
    assert!(read_guest_shared_backing(interior, &mut bytes));
    assert_eq!(&bytes, b"texels");
    assert!(write_guest_shared_backing(interior, b"stored"));
    let mut round_trip = [0_u8; 6];
    memory.read(page(reserved, 1), &mut round_trip).unwrap();
    assert_eq!(&round_trip, b"stored");

    // Distinct committed ranges stay one backing object with virtually-keyed offsets.
    let identity_next = guest_shared_mapping_identity(page(reserved, 2).get(), GUEST_PAGE_SIZE)
        .expect("the rest of the committed range shares the backing");
    assert_eq!(identity_next.backing_id, identity.backing_id);
    assert_eq!(
        identity_next.backing_offset,
        identity.backing_offset + GUEST_PAGE_SIZE
    );
}

#[test]
fn separately_committed_adjacent_pages_resolve_as_one_backing_run() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let reserved = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    // A guest allocator grows a block with a second commit; a GPU resource may then span the
    // seam between the two commit operations.
    memory
        .protect(reserved, 2 * GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();
    memory
        .protect(
            page(reserved, 2),
            2 * GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();

    assert!(
        guest_shared_mapping_identity(reserved.get(), 4 * GUEST_PAGE_SIZE).is_some(),
        "a resource spanning both commits resolves as one physical run"
    );
    assert_eq!(
        guest_shared_backing_segments(reserved.get(), 4 * GUEST_PAGE_SIZE),
        Some(vec![GuestSharedBackingSegment {
            address: reserved.get(),
            byte_count: 4 * GUEST_PAGE_SIZE,
        }]),
        "separate adjacent commits coalesce into one copy/watch segment"
    );
    assert_eq!(
        guest_shared_backing_alias_segments(reserved.get(), 4 * GUEST_PAGE_SIZE),
        Some(vec![GuestSharedBackingSegment {
            address: reserved.get(),
            byte_count: 4 * GUEST_PAGE_SIZE,
        }]),
        "separate adjacent commits coalesce into one write-watch alias"
    );

    // N1: the two separate commits must collapse to exactly one registry record, not one per
    // mprotect. A commit-heavy title otherwise leaves the resolution scans walking one record per
    // page. The backing id is this test's own, so the count is unaffected by concurrent tests.
    let backing_id = guest_shared_mapping_identity(reserved.get(), GUEST_PAGE_SIZE)
        .expect("committed pages have a backing identity")
        .backing_id;
    assert_eq!(
        super::shared::shared_backing_record_count(backing_id),
        1,
        "adjacent contiguous commits collapse to a single coalesced record"
    );
}

#[test]
fn the_backing_index_stays_consistent_across_commit_split_and_release() {
    // N2: the `by_backing` secondary index must track exactly the live records naming a backing as
    // ranges are committed, split by a decommit hole, and released. A stale index entry would make
    // a resolution scan visit a record the address map no longer holds.
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let reserved = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    memory
        .protect(reserved, 4 * GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();
    let backing_id = guest_shared_mapping_identity(reserved.get(), GUEST_PAGE_SIZE)
        .expect("committed pages have a backing identity")
        .backing_id;
    assert_eq!(
        super::shared::shared_backing_record_count(backing_id),
        1,
        "one contiguous commit is one indexed record"
    );

    // Punching a hole in the middle splits the record into two disjoint runs; the index must show
    // both, and a resolution across the hole must fail while each side still resolves.
    memory.unmap(page(reserved, 1), GUEST_PAGE_SIZE).unwrap();
    assert_eq!(
        super::shared::shared_backing_record_count(backing_id),
        2,
        "a decommit hole splits the run into two indexed records"
    );
    assert_eq!(
        guest_shared_mapping_identity(reserved.get(), 4 * GUEST_PAGE_SIZE),
        None,
        "the hole severs resolution across the range"
    );
    assert!(guest_shared_mapping_identity(page(reserved, 2).get(), 2 * GUEST_PAGE_SIZE).is_some());

    // Releasing the whole space retires every record, so the index forgets the backing entirely.
    drop(memory);
    assert_eq!(
        super::shared::shared_backing_record_count(backing_id),
        0,
        "releasing the address space empties the backing index"
    );
}

#[test]
fn decommit_zeroes_pages_and_severs_backing_resolution() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let reserved = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            3 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    memory
        .protect(reserved, 3 * GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();
    assert!(guest_shared_mapping_identity(reserved.get(), 3 * GUEST_PAGE_SIZE).is_some());
    memory.write(page(reserved, 1), b"stale").unwrap();

    memory.unmap(page(reserved, 1), GUEST_PAGE_SIZE).unwrap();
    assert_eq!(
        guest_shared_mapping_identity(reserved.get(), 3 * GUEST_PAGE_SIZE),
        None,
        "a decommitted hole severs resolution across the range"
    );
    assert!(
        guest_shared_mapping_identity(reserved.get(), GUEST_PAGE_SIZE).is_some(),
        "pages on either side of the hole stay resolvable"
    );

    // Recommitting the hole must produce fresh zero pages, never the prior contents.
    memory
        .reserve(
            GuestMapPlacement::Fixed(page(reserved, 1)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    memory
        .protect(
            page(reserved, 1),
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .unwrap();
    let mut recommitted = [0xff_u8; 5];
    memory.read(page(reserved, 1), &mut recommitted).unwrap();
    assert_eq!(&recommitted, &[0; 5], "recommitted pages read as zero");
    assert!(guest_shared_mapping_identity(reserved.get(), 3 * GUEST_PAGE_SIZE).is_some());
}

#[test]
fn reprotecting_committed_pages_preserves_contents_and_identity() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let reserved = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    memory
        .protect(reserved, GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();
    memory.write(reserved, b"kept").unwrap();
    let identity = guest_shared_mapping_identity(reserved.get(), GUEST_PAGE_SIZE)
        .expect("committed pages have a backing identity");

    memory
        .protect(reserved, GUEST_PAGE_SIZE, GuestProtection::READ)
        .unwrap();
    memory
        .protect(reserved, GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();

    assert_eq!(
        guest_shared_mapping_identity(reserved.get(), GUEST_PAGE_SIZE),
        Some(identity),
        "permission changes never reallocate committed backing"
    );
    let mut bytes = [0_u8; 4];
    memory.read(reserved, &mut bytes).unwrap();
    assert_eq!(&bytes, b"kept");
}

#[test]
fn distinct_committed_allocations_never_share_backing_identity() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let first = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    let second = memory
        .reserve(
            GuestMapPlacement::Hint(page(first, 4)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    memory
        .protect(first, GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();
    memory
        .protect(second, GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();

    let first_identity = guest_shared_mapping_identity(first.get(), GUEST_PAGE_SIZE).unwrap();
    let second_identity = guest_shared_mapping_identity(second.get(), GUEST_PAGE_SIZE).unwrap();
    // One space-wide backing object, but virtually-keyed offsets: unrelated committed
    // allocations must never compare as sharing byte storage, because GPU descriptor overlap
    // detection distinguishes resources by (backing id, backing offset).
    assert_eq!(first_identity.backing_id, second_identity.backing_id);
    assert_ne!(
        first_identity.backing_offset,
        second_identity.backing_offset
    );
    assert_eq!(
        second_identity.backing_offset - first_identity.backing_offset,
        second.get() - first.get()
    );
}

#[test]
fn committed_range_watch_observes_stores_through_alias_segments() {
    let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
    let reserved = memory
        .reserve(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
        )
        .unwrap();
    memory
        .protect(reserved, GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
        .unwrap();

    let aliases = guest_shared_backing_alias_segments(reserved.get(), GUEST_PAGE_SIZE)
        .expect("committed pages enumerate their single coherent alias");
    assert_eq!(aliases.len(), 1);
    // SAFETY: the committed mapping stays live until the watch is dropped below.
    let watch = unsafe {
        CpuWriteWatch::new(aliases[0].address, aliases[0].byte_count).expect("alias watch")
    };
    assert!(!watch.is_dirty());
    memory.write(reserved, b"dirty").unwrap();
    assert!(
        watch.is_dirty(),
        "a CPU store through the committed alias is observed"
    );
}

#[test]
fn shared_backing_view_is_coherent_across_disjoint_aliases_and_remaps() {
    let mut memory = GuestAddressSpace::new(10 * GUEST_PAGE_SIZE).unwrap();
    let base = memory.base();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let first = memory
        .map_shared(
            GuestMapPlacement::Fixed(base),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let second = memory
        .map_shared(
            GuestMapPlacement::Fixed(page(base, 4)),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .unwrap();

    let interior_offset = 128_u64;
    let requested_bytes = 2 * GUEST_PAGE_SIZE - 2 * interior_offset;
    let range = guest_shared_backing_range(first.get() + interior_offset, requested_bytes)
        .expect("the physical range resolves through disjoint CPU aliases");
    let two_pages = usize::try_from(2 * GUEST_PAGE_SIZE).expect("two pages fit usize");
    assert_eq!(range.view_identity().backing_id, backing.id());
    assert_eq!(range.view_identity().backing_offset, 0);
    assert_eq!(range.resource_offset(), interior_offset);
    assert_eq!(range.view_byte_count(), two_pages);
    assert!(
        !range.is_virtually_keyed(),
        "direct-memory backing offsets remain durable while the backing is retained"
    );

    let view = range.map_view().expect("map the complete physical view");
    assert_eq!(view.byte_count(), two_pages);
    let second_physical_byte = usize::try_from(range.resource_offset() + GUEST_PAGE_SIZE).unwrap();
    // SAFETY: the mapped view owns two complete backing pages and the selected byte is interior.
    unsafe {
        view.as_ptr()
            .cast_mut()
            .cast::<u8>()
            .add(second_physical_byte)
            .write(0x5a);
    }
    // SAFETY: `second` remains a live writable alias of backing page one.
    assert_eq!(
        unsafe { ((second.get() + interior_offset) as *const u8).read() },
        0x5a
    );

    memory.unmap(first, GUEST_PAGE_SIZE).unwrap();
    memory.unmap(second, GUEST_PAGE_SIZE).unwrap();
    // In-flight users retain the physical mmap independently of retired guest aliases.
    assert_eq!(
        unsafe { view.as_ptr().cast::<u8>().add(second_physical_byte).read() },
        0x5a
    );

    let replacement = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(first),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement,
            0,
        )
        .unwrap();
    let replacement_range =
        guest_shared_backing_range(first.get() + interior_offset, requested_bytes)
            .expect("replacement range resolves");
    assert_ne!(
        replacement_range.view_identity().backing_id,
        range.view_identity().backing_id,
        "a VA remap cannot hit the retired physical-view identity"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn dropping_address_space_releases_exact_range_for_reuse() {
    // Keep this away from host-selected mappings used by concurrently executing tests. Using
    // the address of a dropped anywhere mapping would race another test's mmap allocation.
    let base = GuestAddress::new(0x0000_000a_1000_0000);
    drop(GuestAddressSpace::new_at(base, SPACE_SIZE).unwrap());
    let memory = GuestAddressSpace::new_at(base, SPACE_SIZE).unwrap();
    assert_eq!(memory.base(), base);
}
