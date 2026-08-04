// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

static void remu_copy_exception(NSException *exception, char *error, size_t error_capacity) {
    if (error_capacity == 0) {
        return;
    }
    const char *name = exception.name.UTF8String ?: "NSException";
    const char *reason = exception.reason.UTF8String ?: "no reason";
    snprintf(error, error_capacity, "%s: %s", name, reason);
}

void *remu_new_sampler_argument_encoder(
    id<MTLDevice> device,
    NSUInteger sampler_count,
    char *error,
    size_t error_capacity
) {
    @try {
        NSMutableArray<MTLArgumentDescriptor *> *arguments =
            [NSMutableArray arrayWithCapacity:sampler_count];
        for (NSUInteger index = 0; index < sampler_count; ++index) {
            MTLArgumentDescriptor *descriptor = [[MTLArgumentDescriptor alloc] init];
            descriptor.index = index;
            descriptor.dataType = MTLDataTypeSampler;
            descriptor.arrayLength = 0;
            [arguments addObject:descriptor];
            [descriptor release];
        }
        id<MTLArgumentEncoder> encoder = [device newArgumentEncoderWithArguments:arguments];
        if (encoder == nil && error_capacity != 0) {
            snprintf(error, error_capacity, "Metal returned a null sampler argument encoder");
        }
        return encoder;
    } @catch (NSException *exception) {
        remu_copy_exception(exception, error, error_capacity);
        return NULL;
    }
}

bool remu_argument_encoder_encoded_length(
    id<MTLArgumentEncoder> encoder,
    NSUInteger *length,
    char *error,
    size_t error_capacity
) {
    @try {
        *length = encoder.encodedLength;
        return true;
    } @catch (NSException *exception) {
        remu_copy_exception(exception, error, error_capacity);
        return false;
    }
}

bool remu_argument_encoder_set_argument_buffer(
    id<MTLArgumentEncoder> encoder,
    id<MTLBuffer> buffer,
    NSUInteger offset,
    char *error,
    size_t error_capacity
) {
    @try {
        [encoder setArgumentBuffer:buffer offset:offset];
        return true;
    } @catch (NSException *exception) {
        remu_copy_exception(exception, error, error_capacity);
        return false;
    }
}

void *remu_new_sampler_state(
    id<MTLDevice> device,
    MTLSamplerDescriptor *descriptor,
    char *error,
    size_t error_capacity
) {
    @try {
        id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:descriptor];
        if (sampler == nil && error_capacity != 0) {
            snprintf(error, error_capacity, "Metal returned a null sampler state");
        }
        return sampler;
    } @catch (NSException *exception) {
        remu_copy_exception(exception, error, error_capacity);
        return NULL;
    }
}

bool remu_argument_encoder_set_sampler_state(
    id<MTLArgumentEncoder> encoder,
    id<MTLSamplerState> sampler,
    NSUInteger index,
    char *error,
    size_t error_capacity
) {
    @try {
        [encoder setSamplerState:sampler atIndex:index];
        return true;
    } @catch (NSException *exception) {
        remu_copy_exception(exception, error, error_capacity);
        return false;
    }
}
