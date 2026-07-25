// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Stateful, offline implementations of the NP ABIs used by game executables.
//!
//! `REmu` does not impersonate `PlayStation Network` or upload title telemetry. The universal-data
//! implementation still owns every context, request handle, event, property object and property
//! array exactly as the guest ABI requires. Posting an event commits an immutable local record;
//! destroying the source event afterwards therefore does not discard the completed operation.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const UDS_ERROR_INVALID_ARGUMENT: u32 = 0x8055_3102;
pub const UDS_ERROR_INVALID_CONTEXT: u32 = 0x8055_3104;
pub const UDS_ERROR_INVALID_HANDLE: u32 = 0x8055_3106;
pub const UDS_ERROR_INVALID_EVENT_NAME: u32 = 0x8055_310d;
pub const UDS_ERROR_INVALID_EVENT_PROPERTY_KEY: u32 = 0x8055_310e;
pub const UDS_ERROR_INVALID_EVENT_PROPERTY_VALUE: u32 = 0x8055_310f;
pub const UDS_ERROR_EXCEEDS_EVENT_DATA_SIZE_LIMIT: u32 = 0x8055_3113;
pub const UDS_ERROR_INVALID_EVENT: u32 = 0x8055_3116;
pub const UDS_ERROR_NOT_INITIALIZED: u32 = 0x8055_3117;
pub const UDS_ERROR_ALREADY_INITIALIZED: u32 = 0x8055_3118;
pub const UDS_ERROR_INVALID_OBJECT: u32 = 0x8055_3119;
pub const UDS_ERROR_INVALID_ARRAY: u32 = 0x8055_311a;
pub const UDS_ERROR_NOT_REGISTERED: u32 = 0x8055_3120;
pub const UDS_ERROR_ALREADY_REGISTERED: u32 = 0x8055_3121;

pub const GAME_INTENT_ERROR_ALREADY_INITIALIZED: u32 = 0x8055_3801;
pub const GAME_INTENT_ERROR_NOT_INITIALIZED: u32 = 0x8055_3802;
pub const GAME_INTENT_ERROR_INVALID_ARGUMENT: u32 = 0x8055_3804;
pub const GAME_INTENT_ERROR_INSUFFICIENT_BUFFER: u32 = 0x8055_3805;
pub const GAME_INTENT_ERROR_INTENT_NOT_FOUND: u32 = 0x8055_3806;
pub const GAME_INTENT_ERROR_VALUE_NOT_FOUND: u32 = 0x8055_3807;

pub const TROPHY2_ERROR_INVALID_ARGUMENT: u32 = 0x8055_3904;
pub const TROPHY2_ERROR_ABORT: u32 = 0x8055_3907;
pub const TROPHY2_ERROR_INVALID_HANDLE: u32 = 0x8055_3908;
pub const TROPHY2_ERROR_INVALID_CONTEXT: u32 = 0x8055_3909;

const MAX_EVENT_NAME_BYTES: usize = 255;
const MAX_PROPERTY_KEY_BYTES: usize = 255;
const MAX_PROPERTY_STRING_BYTES: usize = 16 * 1024;
const MAX_EVENT_DATA_BYTES: usize = 64 * 1024;
const MAX_RETAINED_POSTS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
enum EventValue {
    Int32(i32),
    String(String),
    Array(u64),
}

#[derive(Debug, Default)]
struct EventObject {
    properties: BTreeMap<String, EventValue>,
}

#[derive(Debug)]
struct EventArray {
    option: u64,
    values: Vec<EventValue>,
}

#[derive(Debug)]
struct UniversalEvent {
    name: String,
    option: u64,
    root_object: u64,
    objects: BTreeSet<u64>,
    arrays: BTreeSet<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostedEvent {
    pub sequence: u64,
    pub context: i32,
    pub user_id: i32,
    pub context_option: u64,
    pub event_name: String,
    pub event_option: u64,
    pub post_option: u64,
    pub payload_size: usize,
}

#[derive(Debug)]
struct UniversalContext {
    user_id: i32,
    option: u64,
    registered: bool,
}

#[derive(Debug)]
struct UniversalHandle {
    aborted: bool,
}

#[derive(Debug)]
struct PostedEventRecord {
    sequence: u64,
    context: i32,
    event_name: String,
    payload_size: usize,
}

#[derive(Debug, Default)]
pub struct UniversalDataState {
    initialization_parameters: Option<Vec<u8>>,
    next_context: i32,
    contexts: BTreeMap<i32, UniversalContext>,
    next_handle: i32,
    handles: BTreeMap<i32, UniversalHandle>,
    next_event: u64,
    events: BTreeMap<u64, UniversalEvent>,
    next_property: u64,
    objects: BTreeMap<u64, EventObject>,
    arrays: BTreeMap<u64, EventArray>,
    next_post_sequence: u64,
    posted: VecDeque<PostedEventRecord>,
}

impl UniversalDataState {
    pub fn initialize(&mut self, parameters: Vec<u8>) -> Result<(), u32> {
        if parameters.is_empty() {
            return Err(UDS_ERROR_INVALID_ARGUMENT);
        }
        if self.initialization_parameters.is_some() {
            return Err(UDS_ERROR_ALREADY_INITIALIZED);
        }
        self.initialization_parameters = Some(parameters);
        self.next_context = 1;
        self.next_handle = 1;
        self.next_event = 1;
        self.next_property = 1;
        self.next_post_sequence = 1;
        Ok(())
    }

    pub fn terminate(&mut self) -> Result<(), u32> {
        self.require_initialized()?;
        *self = Self::default();
        Ok(())
    }

    pub fn create_context(&mut self, user_id: i32, option: u64) -> Result<i32, u32> {
        self.require_initialized()?;
        let context = allocate_i32_id(&mut self.next_context);
        self.contexts.insert(
            context,
            UniversalContext {
                user_id,
                option,
                registered: false,
            },
        );
        Ok(context)
    }

    pub fn destroy_context(&mut self, context: i32) -> Result<(), u32> {
        self.require_initialized()?;
        self.contexts
            .remove(&context)
            .map(|_| ())
            .ok_or(UDS_ERROR_INVALID_CONTEXT)
    }

    pub fn create_handle(&mut self) -> Result<i32, u32> {
        self.require_initialized()?;
        let handle = allocate_i32_id(&mut self.next_handle);
        self.handles
            .insert(handle, UniversalHandle { aborted: false });
        Ok(handle)
    }

    pub fn destroy_handle(&mut self, handle: i32) -> Result<(), u32> {
        self.require_initialized()?;
        self.handles
            .remove(&handle)
            .map(|_| ())
            .ok_or(UDS_ERROR_INVALID_HANDLE)
    }

    pub fn abort_handle(&mut self, handle: i32) -> Result<(), u32> {
        self.require_initialized()?;
        let handle = self
            .handles
            .get_mut(&handle)
            .ok_or(UDS_ERROR_INVALID_HANDLE)?;
        handle.aborted = true;
        Ok(())
    }

    pub fn register_context(&mut self, context: i32, handle: i32) -> Result<(), u32> {
        self.require_initialized()?;
        let request = self.handles.get(&handle).ok_or(UDS_ERROR_INVALID_HANDLE)?;
        if request.aborted {
            return Err(UDS_ERROR_INVALID_HANDLE);
        }
        let context = self
            .contexts
            .get_mut(&context)
            .ok_or(UDS_ERROR_INVALID_CONTEXT)?;
        if context.registered {
            return Err(UDS_ERROR_ALREADY_REGISTERED);
        }
        context.registered = true;
        Ok(())
    }

    pub fn create_event(&mut self, name: String, option: u64) -> Result<(u64, u64), u32> {
        self.require_initialized()?;
        validate_event_name(&name)?;
        let event = allocate_u64_id(&mut self.next_event);
        let root_object = allocate_u64_id(&mut self.next_property);
        self.objects.insert(root_object, EventObject::default());
        self.events.insert(
            event,
            UniversalEvent {
                name,
                option,
                root_object,
                objects: BTreeSet::from([root_object]),
                arrays: BTreeSet::new(),
            },
        );
        Ok((event, root_object))
    }

    pub fn destroy_event(&mut self, event: u64) -> Result<(), u32> {
        self.require_initialized()?;
        let event = self.events.remove(&event).ok_or(UDS_ERROR_INVALID_EVENT)?;
        for object in event.objects {
            self.objects.remove(&object);
        }
        for array in event.arrays {
            self.arrays.remove(&array);
        }
        Ok(())
    }

    pub fn create_property_object(&mut self) -> Result<u64, u32> {
        self.require_initialized()?;
        let object = allocate_u64_id(&mut self.next_property);
        self.objects.insert(object, EventObject::default());
        Ok(object)
    }

    pub fn object_set_i32(&mut self, object: u64, key: String, value: i32) -> Result<(), u32> {
        validate_property_key(&key)?;
        self.object_set_value(object, key, EventValue::Int32(value))
    }

    pub fn object_set_string(
        &mut self,
        object: u64,
        key: String,
        value: String,
    ) -> Result<(), u32> {
        validate_property_key(&key)?;
        validate_property_string(&value)?;
        self.object_set_value(object, key, EventValue::String(value))
    }

    pub fn object_set_array(&mut self, object: u64, key: String, option: u64) -> Result<u64, u32> {
        validate_property_key(&key)?;
        let owner = self
            .event_owning_object(object)
            .ok_or(UDS_ERROR_INVALID_OBJECT)?;
        let array = allocate_u64_id(&mut self.next_property);
        let property_key = key.clone();
        self.arrays.insert(
            array,
            EventArray {
                option,
                values: Vec::new(),
            },
        );
        let previous = self
            .objects
            .get_mut(&object)
            .expect("the owner lookup established a live property object")
            .properties
            .insert(key, EventValue::Array(array));
        self.events
            .get_mut(&owner)
            .expect("the owner lookup established a live event")
            .arrays
            .insert(array);
        if let Err(error) = self.validate_event_size(owner) {
            let object = self
                .objects
                .get_mut(&object)
                .expect("the property object remains live after a size rejection");
            if let Some(previous) = previous {
                object.properties.insert(property_key, previous);
            } else {
                object.properties.remove(&property_key);
            }
            self.arrays.remove(&array);
            self.events
                .get_mut(&owner)
                .expect("the event remains live after a size rejection")
                .arrays
                .remove(&array);
            return Err(error);
        }
        if let Some(EventValue::Array(previous_array)) = previous {
            self.arrays.remove(&previous_array);
            self.events
                .get_mut(&owner)
                .expect("the owner event remains live after replacing an array")
                .arrays
                .remove(&previous_array);
        }
        Ok(array)
    }

    pub fn array_append_string(&mut self, array: u64, value: String) -> Result<(), u32> {
        validate_property_string(&value)?;
        let owner = self
            .event_owning_array(array)
            .ok_or(UDS_ERROR_INVALID_ARRAY)?;
        self.arrays
            .get_mut(&array)
            .expect("the owner lookup established a live property array")
            .values
            .push(EventValue::String(value));
        if let Err(error) = self.validate_event_size(owner) {
            self.arrays
                .get_mut(&array)
                .expect("the array remains live after a size rejection")
                .values
                .pop();
            return Err(error);
        }
        Ok(())
    }

    pub fn post_event(
        &mut self,
        context: i32,
        handle: i32,
        event: u64,
        option: u64,
    ) -> Result<PostedEvent, u32> {
        self.require_initialized()?;
        let context_state = self
            .contexts
            .get(&context)
            .ok_or(UDS_ERROR_INVALID_CONTEXT)?;
        if !context_state.registered {
            return Err(UDS_ERROR_NOT_REGISTERED);
        }
        let request = self.handles.get(&handle).ok_or(UDS_ERROR_INVALID_HANDLE)?;
        if request.aborted {
            return Err(UDS_ERROR_INVALID_HANDLE);
        }
        let payload_size = self.event_size(event)?;
        let source = self.events.get(&event).ok_or(UDS_ERROR_INVALID_EVENT)?;
        // Retain these observed ABI inputs in the committed record. A future online transport can
        // consume the same records without changing guest-visible ownership semantics.
        let user_id = context_state.user_id;
        let context_option = context_state.option;
        let event_name = source.name.clone();
        let event_option = source.option;
        let sequence = allocate_u64_id(&mut self.next_post_sequence);
        self.posted.push_back(PostedEventRecord {
            sequence,
            context,
            event_name: source.name.clone(),
            payload_size,
        });
        if self.posted.len() > MAX_RETAINED_POSTS {
            self.posted.pop_front();
        }
        let record = self
            .posted
            .back()
            .expect("the posted event was inserted immediately above");
        debug_assert_eq!(record.sequence, sequence);
        debug_assert_eq!(record.context, context);
        debug_assert_eq!(record.event_name, source.name);
        debug_assert_eq!(record.payload_size, payload_size);
        Ok(PostedEvent {
            sequence,
            context,
            user_id,
            context_option,
            event_name,
            event_option,
            post_option: option,
            payload_size,
        })
    }

    fn require_initialized(&self) -> Result<(), u32> {
        self.initialization_parameters
            .as_ref()
            .map(|_| ())
            .ok_or(UDS_ERROR_NOT_INITIALIZED)
    }

    fn object_mut(&mut self, object: u64) -> Result<&mut EventObject, u32> {
        self.require_initialized()?;
        self.objects
            .get_mut(&object)
            .ok_or(UDS_ERROR_INVALID_OBJECT)
    }

    fn object_set_value(&mut self, object: u64, key: String, value: EventValue) -> Result<(), u32> {
        let owner = self
            .event_owning_object(object)
            .ok_or(UDS_ERROR_INVALID_OBJECT)?;
        let previous = self
            .object_mut(object)?
            .properties
            .insert(key.clone(), value);
        if let Err(error) = self.validate_owner_size(object) {
            let properties = &mut self
                .objects
                .get_mut(&object)
                .expect("the property object remains live after a size rejection")
                .properties;
            if let Some(previous) = previous {
                properties.insert(key, previous);
            } else {
                properties.remove(&key);
            }
            return Err(error);
        }
        if let Some(EventValue::Array(previous_array)) = previous {
            self.arrays.remove(&previous_array);
            self.events
                .get_mut(&owner)
                .expect("the owner event remains live after replacing an array")
                .arrays
                .remove(&previous_array);
        }
        Ok(())
    }

    fn event_owning_object(&self, object: u64) -> Option<u64> {
        self.events
            .iter()
            .find_map(|(id, event)| event.objects.contains(&object).then_some(*id))
    }

    fn event_owning_array(&self, array: u64) -> Option<u64> {
        self.events
            .iter()
            .find_map(|(id, event)| event.arrays.contains(&array).then_some(*id))
    }

    fn validate_owner_size(&self, object: u64) -> Result<(), u32> {
        let owner = self
            .event_owning_object(object)
            .ok_or(UDS_ERROR_INVALID_OBJECT)?;
        self.validate_event_size(owner)
    }

    fn validate_event_size(&self, event: u64) -> Result<(), u32> {
        if self.event_size(event)? > MAX_EVENT_DATA_BYTES {
            Err(UDS_ERROR_EXCEEDS_EVENT_DATA_SIZE_LIMIT)
        } else {
            Ok(())
        }
    }

    fn event_size(&self, event: u64) -> Result<usize, u32> {
        let event = self.events.get(&event).ok_or(UDS_ERROR_INVALID_EVENT)?;
        let object = self
            .objects
            .get(&event.root_object)
            .ok_or(UDS_ERROR_INVALID_OBJECT)?;
        Ok(event.name.len() + encoded_object_size(object, &self.arrays))
    }
}

fn encoded_object_size(object: &EventObject, arrays: &BTreeMap<u64, EventArray>) -> usize {
    object.properties.iter().fold(0, |size, (key, value)| {
        size.saturating_add(key.len())
            .saturating_add(encoded_value_size(value, arrays))
    })
}

fn encoded_value_size(value: &EventValue, arrays: &BTreeMap<u64, EventArray>) -> usize {
    match value {
        EventValue::Int32(value) => {
            let _ = value;
            std::mem::size_of::<i32>()
        }
        EventValue::String(value) => value.len(),
        EventValue::Array(array) => arrays.get(array).map_or(0, |array| {
            let option_size = usize::from(array.option != 0) * std::mem::size_of::<u64>();
            array.values.iter().fold(option_size, |size, value| {
                size.saturating_add(encoded_value_size(value, arrays))
            })
        }),
    }
}

fn validate_event_name(name: &str) -> Result<(), u32> {
    if name.is_empty() || name.len() > MAX_EVENT_NAME_BYTES || name.as_bytes().contains(&0) {
        Err(UDS_ERROR_INVALID_EVENT_NAME)
    } else {
        Ok(())
    }
}

fn validate_property_key(key: &str) -> Result<(), u32> {
    if key.is_empty() || key.len() > MAX_PROPERTY_KEY_BYTES || key.as_bytes().contains(&0) {
        Err(UDS_ERROR_INVALID_EVENT_PROPERTY_KEY)
    } else {
        Ok(())
    }
}

fn validate_property_string(value: &str) -> Result<(), u32> {
    if value.len() > MAX_PROPERTY_STRING_BYTES || value.as_bytes().contains(&0) {
        Err(UDS_ERROR_INVALID_EVENT_PROPERTY_VALUE)
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameIntent {
    pub intent_type: String,
    pub properties: BTreeMap<String, String>,
}

#[derive(Debug, Default)]
pub struct GameIntentState {
    initialization_parameters: Option<Vec<u8>>,
    pending: VecDeque<GameIntent>,
    received_properties: BTreeMap<u64, BTreeMap<String, String>>,
}

impl GameIntentState {
    pub fn initialize(&mut self, parameters: Vec<u8>) -> Result<(), u32> {
        if parameters.is_empty() {
            return Err(GAME_INTENT_ERROR_INVALID_ARGUMENT);
        }
        if self.initialization_parameters.is_some() {
            return Err(GAME_INTENT_ERROR_ALREADY_INITIALIZED);
        }
        self.initialization_parameters = Some(parameters);
        Ok(())
    }

    pub fn receive(&mut self) -> Result<GameIntent, u32> {
        if self.initialization_parameters.is_none() {
            return Err(GAME_INTENT_ERROR_NOT_INITIALIZED);
        }
        self.pending
            .pop_front()
            .ok_or(GAME_INTENT_ERROR_INTENT_NOT_FOUND)
    }

    pub fn retain_received_properties(
        &mut self,
        property_object: u64,
        properties: BTreeMap<String, String>,
    ) {
        self.received_properties.insert(property_object, properties);
    }

    pub fn property(&self, property_object: u64, key: &str) -> Result<&str, u32> {
        if self.initialization_parameters.is_none() {
            return Err(GAME_INTENT_ERROR_NOT_INITIALIZED);
        }
        self.received_properties
            .get(&property_object)
            .and_then(|properties| properties.get(key))
            .map(String::as_str)
            .ok_or(GAME_INTENT_ERROR_VALUE_NOT_FOUND)
    }

    #[cfg(test)]
    pub fn enqueue(&mut self, intent: GameIntent) {
        self.pending.push_back(intent);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrophyUnlockCallback {
    pub function: u64,
    pub argument: u64,
}

#[derive(Debug)]
struct TrophyContext {
    user_id: i32,
    option: u64,
    registered: bool,
}

#[derive(Debug)]
struct TrophyHandle {
    aborted: bool,
}

#[derive(Debug, Default)]
pub struct TrophyState {
    next_context: i32,
    contexts: BTreeMap<i32, TrophyContext>,
    next_handle: i32,
    handles: BTreeMap<i32, TrophyHandle>,
    unlock_callback: Option<TrophyUnlockCallback>,
}

impl TrophyState {
    pub fn create_context(&mut self, user_id: i32, option: u64) -> i32 {
        let context = allocate_i32_id(&mut self.next_context);
        self.contexts.insert(
            context,
            TrophyContext {
                user_id,
                option,
                registered: false,
            },
        );
        context
    }

    pub fn destroy_context(&mut self, context: i32) -> Result<(), u32> {
        self.contexts
            .remove(&context)
            .map(|_| ())
            .ok_or(TROPHY2_ERROR_INVALID_CONTEXT)
    }

    pub fn create_handle(&mut self) -> i32 {
        let handle = allocate_i32_id(&mut self.next_handle);
        self.handles.insert(handle, TrophyHandle { aborted: false });
        handle
    }

    pub fn destroy_handle(&mut self, handle: i32) -> Result<(), u32> {
        self.handles
            .remove(&handle)
            .map(|_| ())
            .ok_or(TROPHY2_ERROR_INVALID_HANDLE)
    }

    pub fn abort_handle(&mut self, handle: i32) -> Result<(), u32> {
        let handle = self
            .handles
            .get_mut(&handle)
            .ok_or(TROPHY2_ERROR_INVALID_HANDLE)?;
        handle.aborted = true;
        Ok(())
    }

    pub fn register_context(&mut self, context: i32, handle: i32) -> Result<(), u32> {
        let handle_state = self
            .handles
            .get(&handle)
            .ok_or(TROPHY2_ERROR_INVALID_HANDLE)?;
        if handle_state.aborted {
            return Err(TROPHY2_ERROR_ABORT);
        }
        let context_id = context;
        let context = self
            .contexts
            .get_mut(&context)
            .ok_or(TROPHY2_ERROR_INVALID_CONTEXT)?;
        eprintln!(
            "[REMu][TROPHY] registered context={context_id} user={:#x} option={:#x}",
            context.user_id, context.option
        );
        context.registered = true;
        Ok(())
    }

    pub fn register_unlock_callback(&mut self, function: u64, argument: u64) -> Result<(), u32> {
        if function == 0 {
            return Err(TROPHY2_ERROR_INVALID_ARGUMENT);
        }
        self.unlock_callback = Some(TrophyUnlockCallback { function, argument });
        Ok(())
    }

    pub fn unregister_unlock_callback(&mut self) -> Result<(), u32> {
        self.unlock_callback
            .take()
            .map(|_| ())
            .ok_or(TROPHY2_ERROR_INVALID_ARGUMENT)
    }
}

fn allocate_i32_id(next: &mut i32) -> i32 {
    let id = (*next).max(1);
    *next = id.checked_add(1).unwrap_or(1);
    id
}

fn allocate_u64_id(next: &mut u64) -> u64 {
    let id = (*next).max(1);
    *next = id.checked_add(1).unwrap_or(1);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn universal_data_owns_a_complete_event_graph_until_destruction() {
        let mut state = UniversalDataState::default();
        state.initialize(vec![16; 16]).unwrap();
        let context = state.create_context(0x1000_0000, 0).unwrap();
        let registration_handle = state.create_handle().unwrap();
        state
            .register_context(context, registration_handle)
            .unwrap();
        state.destroy_handle(registration_handle).unwrap();

        let request = state.create_handle().unwrap();
        let (event, root) = state.create_event("levelComplete".to_owned(), 0).unwrap();
        state.object_set_i32(root, "level".to_owned(), 7).unwrap();
        state
            .object_set_string(root, "result".to_owned(), "win".to_owned())
            .unwrap();
        let array = state
            .object_set_array(root, "collectibles".to_owned(), 0)
            .unwrap();
        state.array_append_string(array, "key".to_owned()).unwrap();
        let posted = state.post_event(context, request, event, 0).unwrap();
        assert_eq!(posted.sequence, 1);
        assert!(posted.payload_size >= "levelCompleteresultwincollectibleskey".len());
        state.destroy_handle(request).unwrap();
        state.destroy_event(event).unwrap();
        assert_eq!(
            state.object_set_i32(root, "level".to_owned(), 8),
            Err(UDS_ERROR_INVALID_OBJECT)
        );
        assert_eq!(
            state.array_append_string(array, "coin".to_owned()),
            Err(UDS_ERROR_INVALID_ARRAY)
        );
    }

    #[test]
    fn universal_data_rejects_unregistered_and_aborted_requests() {
        let mut state = UniversalDataState::default();
        state.initialize(vec![0; 16]).unwrap();
        let context = state.create_context(1, 0).unwrap();
        let handle = state.create_handle().unwrap();
        let (event, _) = state.create_event("boot".to_owned(), 0).unwrap();
        assert_eq!(
            state.post_event(context, handle, event, 0),
            Err(UDS_ERROR_NOT_REGISTERED)
        );
        state.register_context(context, handle).unwrap();
        state.abort_handle(handle).unwrap();
        assert_eq!(
            state.post_event(context, handle, event, 0),
            Err(UDS_ERROR_INVALID_HANDLE)
        );
    }

    #[test]
    fn game_intents_retain_properties_for_the_getter_abi() {
        let mut state = GameIntentState::default();
        state.initialize(vec![0; 0x28]).unwrap();
        state.enqueue(GameIntent {
            intent_type: "launchActivity".to_owned(),
            properties: BTreeMap::from([("activityId".to_owned(), "intro".to_owned())]),
        });
        let intent = state.receive().unwrap();
        state.retain_received_properties(0x1234, intent.properties);
        assert_eq!(state.property(0x1234, "activityId"), Ok("intro"));
        assert_eq!(
            state.property(0x1234, "missing"),
            Err(GAME_INTENT_ERROR_VALUE_NOT_FOUND)
        );
    }

    #[test]
    fn trophy_lifetimes_validate_handles_and_callbacks() {
        let mut state = TrophyState::default();
        assert_eq!(
            state.register_unlock_callback(0, 0),
            Err(TROPHY2_ERROR_INVALID_ARGUMENT)
        );
        state.register_unlock_callback(0x1234, 0x5678).unwrap();
        let context = state.create_context(1, 0);
        let handle = state.create_handle();
        state.register_context(context, handle).unwrap();
        state.destroy_handle(handle).unwrap();
        assert_eq!(
            state.destroy_handle(handle),
            Err(TROPHY2_ERROR_INVALID_HANDLE)
        );
        state.unregister_unlock_callback().unwrap();
    }
}
