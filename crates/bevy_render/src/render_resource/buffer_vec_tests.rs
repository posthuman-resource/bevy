use super::*;

#[test]
fn read_only_upload_tracks_exact_contents_and_buffer_identity() {
    let first = BufferId::new();
    let replacement = BufferId::new();
    let mut buffer = RawBufferVec::<u32>::new(BufferUsages::STORAGE);
    buffer.push(42);
    assert!(!buffer.upload_is_current(first));
    buffer.remember_upload(first);
    assert!(buffer.upload_is_current(first));

    // A no-op CPU rebuild has identical bytes and needs no staging upload.
    buffer.clear();
    buffer.push(42);
    assert!(buffer.upload_is_current(first));

    buffer.values_mut()[0] = 43;
    assert!(!buffer.upload_is_current(first));
    buffer.remember_upload(first);
    assert!(buffer.upload_is_current(first));

    // A reallocated GPU buffer has no initialized contents, even when its
    // size and the source bytes agree with the previous allocation.
    assert!(!buffer.upload_is_current(replacement));
    buffer.remember_upload(replacement);
    assert!(buffer.upload_is_current(replacement));
}

#[test]
fn read_only_upload_detects_length_changes_and_restore() {
    let id = BufferId::new();
    let mut buffer = RawBufferVec::<u32>::new(BufferUsages::STORAGE);
    buffer.extend([10, 20]);
    buffer.remember_upload(id);
    buffer.truncate(1);
    assert!(!buffer.upload_is_current(id));
    buffer.remember_upload(id);
    buffer.push(20);
    assert!(!buffer.upload_is_current(id));
    buffer.remember_upload(id);
    assert!(buffer.upload_is_current(id));
    buffer.uploaded_bytes = None;
    assert!(!buffer.upload_is_current(id));
}
