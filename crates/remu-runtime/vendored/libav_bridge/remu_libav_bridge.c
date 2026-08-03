/* Copyright (C) 2026 REmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * In-process libav* decode for AvPlayer + Videodec2 (MASTER-PLAN P5).
 * When REMU_LIBAV_STUB is defined the symbols exist but refuse work loudly via return codes.
 */

#include "remu_libav_bridge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef REMU_LIBAV_STUB

int remu_libav_available(void) { return 0; }

int remu_libav_probe_video(
    const char *path,
    int32_t *width_out,
    int32_t *height_out,
    double *fps_out,
    double *duration_out) {
    (void)path;
    (void)width_out;
    (void)height_out;
    (void)fps_out;
    (void)duration_out;
    return 0;
}

int remu_libav_probe_has_audio(const char *path) {
    (void)path;
    return 0;
}

RemuLibavDemux *remu_libav_demux_open(
    const char *path,
    int audio,
    double seek_seconds,
    char *err_buf,
    size_t err_buf_len) {
    (void)path;
    (void)audio;
    (void)seek_seconds;
    if (err_buf && err_buf_len) {
        snprintf(err_buf, err_buf_len, "libav bridge stub (no system ffmpeg for this target)");
    }
    return NULL;
}

int remu_libav_demux_read_frame(RemuLibavDemux *session, uint8_t *out, size_t out_len) {
    (void)session;
    (void)out;
    (void)out_len;
    return -1;
}

void remu_libav_demux_close(RemuLibavDemux *session) { (void)session; }

RemuLibavCodec *remu_libav_codec_open(uint32_t codec_id, char *err_buf, size_t err_buf_len) {
    (void)codec_id;
    if (err_buf && err_buf_len) {
        snprintf(err_buf, err_buf_len, "libav bridge stub (no system ffmpeg for this target)");
    }
    return NULL;
}

int remu_libav_codec_send(
    RemuLibavCodec *codec,
    const uint8_t *au,
    size_t au_size,
    int64_t pts) {
    (void)codec;
    (void)au;
    (void)au_size;
    (void)pts;
    return -1;
}

int remu_libav_codec_receive_nv12(
    RemuLibavCodec *codec,
    uint8_t *out,
    size_t out_len,
    int32_t *width_out,
    int32_t *height_out,
    int32_t *pitch_out) {
    (void)codec;
    (void)out;
    (void)out_len;
    (void)width_out;
    (void)height_out;
    (void)pitch_out;
    return -1;
}

int remu_libav_codec_flush(RemuLibavCodec *codec) {
    (void)codec;
    return -1;
}

void remu_libav_codec_close(RemuLibavCodec *codec) { (void)codec; }

#else /* real libav* */

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>

struct RemuLibavDemux {
    AVFormatContext *fmt;
    AVCodecContext *codec;
    SwsContext *sws;
    SwrContext *swr;
    AVFrame *frame;
    AVFrame *nv12;
    AVPacket *packet;
    int stream_index;
    int audio;
    int width;
    int height;
    int eof;
    uint8_t *audio_staging;
    int audio_staging_samples; /* stereo frames queued */
    int audio_staging_capacity;
};

struct RemuLibavCodec {
    AVCodecContext *codec;
    SwsContext *sws;
    AVFrame *frame;
    AVFrame *nv12;
    AVPacket *packet;
    int width;
    int height;
};

static void set_err(char *err_buf, size_t err_buf_len, const char *msg) {
    if (err_buf && err_buf_len) {
        snprintf(err_buf, err_buf_len, "%s", msg ? msg : "libav error");
    }
}

int remu_libav_available(void) { return 1; }

static int open_decoder_for_stream(
    AVFormatContext *fmt,
    int stream_index,
    AVCodecContext **codec_out,
    char *err_buf,
    size_t err_buf_len) {
    const AVCodecParameters *params = fmt->streams[stream_index]->codecpar;
    const AVCodec *decoder = avcodec_find_decoder(params->codec_id);
    if (!decoder) {
        set_err(err_buf, err_buf_len, "no decoder for stream codec");
        return -1;
    }
    AVCodecContext *codec = avcodec_alloc_context3(decoder);
    if (!codec) {
        set_err(err_buf, err_buf_len, "avcodec_alloc_context3 failed");
        return -1;
    }
    if (avcodec_parameters_to_context(codec, params) < 0) {
        set_err(err_buf, err_buf_len, "avcodec_parameters_to_context failed");
        avcodec_free_context(&codec);
        return -1;
    }
    if (avcodec_open2(codec, decoder, NULL) < 0) {
        set_err(err_buf, err_buf_len, "avcodec_open2 failed");
        avcodec_free_context(&codec);
        return -1;
    }
    *codec_out = codec;
    return 0;
}

int remu_libav_probe_video(
    const char *path,
    int32_t *width_out,
    int32_t *height_out,
    double *fps_out,
    double *duration_out) {
    if (!path || !width_out || !height_out || !fps_out || !duration_out) {
        return 0;
    }
    AVFormatContext *fmt = NULL;
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) {
        return 0;
    }
    if (avformat_find_stream_info(fmt, NULL) < 0) {
        avformat_close_input(&fmt);
        return 0;
    }
    int stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (stream < 0) {
        avformat_close_input(&fmt);
        return 0;
    }
    AVStream *st = fmt->streams[stream];
    *width_out = st->codecpar->width;
    *height_out = st->codecpar->height;
    AVRational rate = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    *fps_out = (rate.den != 0) ? av_q2d(rate) : 30.0;
    if (*fps_out <= 0.0 || !isfinite(*fps_out)) {
        *fps_out = 30.0;
    }
    double duration = 0.0;
    if (st->duration > 0 && st->time_base.den) {
        duration = st->duration * av_q2d(st->time_base);
    } else if (fmt->duration > 0) {
        duration = (double)fmt->duration / (double)AV_TIME_BASE;
    }
    *duration_out = duration > 0.0 ? duration : 0.0;
    avformat_close_input(&fmt);
    return (*width_out > 0 && *height_out > 0) ? 1 : 0;
}

int remu_libav_probe_has_audio(const char *path) {
    if (!path) {
        return 0;
    }
    AVFormatContext *fmt = NULL;
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) {
        return 0;
    }
    if (avformat_find_stream_info(fmt, NULL) < 0) {
        avformat_close_input(&fmt);
        return 0;
    }
    int stream = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    avformat_close_input(&fmt);
    return stream >= 0 ? 1 : 0;
}

static int copy_nv12_from_frame(AVFrame *src, uint8_t *out, size_t out_len, int width, int height) {
    size_t need = (size_t)width * (size_t)height * 3u / 2u;
    if (out_len < need) {
        return -1;
    }
    /* Tightly packed NV12: Y plane then interleaved UV. */
    for (int y = 0; y < height; y++) {
        memcpy(out + (size_t)y * (size_t)width, src->data[0] + y * src->linesize[0], (size_t)width);
    }
    uint8_t *uv = out + (size_t)width * (size_t)height;
    for (int y = 0; y < (height + 1) / 2; y++) {
        memcpy(uv + (size_t)y * (size_t)width, src->data[1] + y * src->linesize[1], (size_t)width);
    }
    return 0;
}

static int ensure_nv12_converter(RemuLibavDemux *session) {
    if (session->sws && session->nv12
        && session->nv12->width == session->width
        && session->nv12->height == session->height) {
        return 0;
    }
    if (session->sws) {
        sws_freeContext(session->sws);
        session->sws = NULL;
    }
    if (session->nv12) {
        av_frame_free(&session->nv12);
    }
    session->nv12 = av_frame_alloc();
    if (!session->nv12) {
        return -1;
    }
    session->nv12->format = AV_PIX_FMT_NV12;
    session->nv12->width = session->width;
    session->nv12->height = session->height;
    if (av_frame_get_buffer(session->nv12, 32) < 0) {
        return -1;
    }
    session->sws = sws_getContext(
        session->width,
        session->height,
        session->codec->pix_fmt,
        session->width,
        session->height,
        AV_PIX_FMT_NV12,
        SWS_BILINEAR,
        NULL,
        NULL,
        NULL);
    return session->sws ? 0 : -1;
}

static int write_video_frame(RemuLibavDemux *session, uint8_t *out, size_t out_len) {
    if (session->frame->format == AV_PIX_FMT_NV12) {
        return copy_nv12_from_frame(
            session->frame, out, out_len, session->width, session->height);
    }
    if (ensure_nv12_converter(session) < 0) {
        return -1;
    }
    if (sws_scale(
            session->sws,
            (const uint8_t *const *)session->frame->data,
            session->frame->linesize,
            0,
            session->height,
            session->nv12->data,
            session->nv12->linesize)
        <= 0) {
        return -1;
    }
    return copy_nv12_from_frame(session->nv12, out, out_len, session->width, session->height);
}

static int ensure_swr(RemuLibavDemux *session) {
    if (session->swr) {
        return 0;
    }
    session->swr = swr_alloc();
    if (!session->swr) {
        return -1;
    }
    av_opt_set_chlayout(session->swr, "in_chlayout", &session->codec->ch_layout, 0);
    av_opt_set_int(session->swr, "in_sample_rate", session->codec->sample_rate, 0);
    av_opt_set_sample_fmt(session->swr, "in_sample_fmt", session->codec->sample_fmt, 0);
    AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
    av_opt_set_chlayout(session->swr, "out_chlayout", &stereo, 0);
    av_opt_set_int(session->swr, "out_sample_rate", 48000, 0);
    av_opt_set_sample_fmt(session->swr, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
    if (swr_init(session->swr) < 0) {
        swr_free(&session->swr);
        return -1;
    }
    return 0;
}

static int append_audio_samples(RemuLibavDemux *session, const uint8_t *samples, int frames) {
    int need = session->audio_staging_samples + frames;
    if (need > session->audio_staging_capacity) {
        int capacity = need < 4096 ? 4096 : need * 2;
        uint8_t *grown = (uint8_t *)realloc(session->audio_staging, (size_t)capacity * 4u);
        if (!grown) {
            return -1;
        }
        session->audio_staging = grown;
        session->audio_staging_capacity = capacity;
    }
    memcpy(
        session->audio_staging + (size_t)session->audio_staging_samples * 4u,
        samples,
        (size_t)frames * 4u);
    session->audio_staging_samples += frames;
    return 0;
}

static int write_audio_grain(RemuLibavDemux *session, uint8_t *out, size_t out_len) {
    /* Match the historical ffmpeg pipe grain: 1024 stereo s16le frames = 4096 bytes. */
    const int grain = 1024;
    if (out_len < (size_t)grain * 4u) {
        return -1;
    }
    while (session->audio_staging_samples < grain && !session->eof) {
        int ret = av_read_frame(session->fmt, session->packet);
        if (ret < 0) {
            session->eof = 1;
            break;
        }
        if (session->packet->stream_index != session->stream_index) {
            av_packet_unref(session->packet);
            continue;
        }
        if (avcodec_send_packet(session->codec, session->packet) < 0) {
            av_packet_unref(session->packet);
            continue;
        }
        av_packet_unref(session->packet);
        while (avcodec_receive_frame(session->codec, session->frame) == 0) {
            if (ensure_swr(session) < 0) {
                return -1;
            }
            int out_count = swr_get_out_samples(session->swr, session->frame->nb_samples);
            if (out_count <= 0) {
                continue;
            }
            uint8_t *converted = (uint8_t *)malloc((size_t)out_count * 4u);
            if (!converted) {
                return -1;
            }
            uint8_t *out_planes[1] = {converted};
            int produced = swr_convert(
                session->swr,
                out_planes,
                out_count,
                (const uint8_t **)session->frame->extended_data,
                session->frame->nb_samples);
            if (produced > 0) {
                if (append_audio_samples(session, converted, produced) < 0) {
                    free(converted);
                    return -1;
                }
            }
            free(converted);
        }
    }
    if (session->audio_staging_samples < grain) {
        if (session->audio_staging_samples == 0) {
            return 1; /* EOF */
        }
        /* Pad a short final grain with silence so the consumer sees a full frame. */
        memset(
            session->audio_staging + (size_t)session->audio_staging_samples * 4u,
            0,
            (size_t)(grain - session->audio_staging_samples) * 4u);
        session->audio_staging_samples = grain;
    }
    memcpy(out, session->audio_staging, (size_t)grain * 4u);
    session->audio_staging_samples -= grain;
    if (session->audio_staging_samples > 0) {
        memmove(
            session->audio_staging,
            session->audio_staging + (size_t)grain * 4u,
            (size_t)session->audio_staging_samples * 4u);
    }
    return 0;
}

RemuLibavDemux *remu_libav_demux_open(
    const char *path,
    int audio,
    double seek_seconds,
    char *err_buf,
    size_t err_buf_len) {
    if (!path) {
        set_err(err_buf, err_buf_len, "null path");
        return NULL;
    }
    RemuLibavDemux *session = (RemuLibavDemux *)calloc(1, sizeof(*session));
    if (!session) {
        set_err(err_buf, err_buf_len, "oom");
        return NULL;
    }
    session->audio = audio ? 1 : 0;
    session->stream_index = -1;
    if (avformat_open_input(&session->fmt, path, NULL, NULL) < 0) {
        set_err(err_buf, err_buf_len, "avformat_open_input failed");
        free(session);
        return NULL;
    }
    if (avformat_find_stream_info(session->fmt, NULL) < 0) {
        set_err(err_buf, err_buf_len, "avformat_find_stream_info failed");
        remu_libav_demux_close(session);
        return NULL;
    }
    enum AVMediaType type = audio ? AVMEDIA_TYPE_AUDIO : AVMEDIA_TYPE_VIDEO;
    int stream = av_find_best_stream(session->fmt, type, -1, -1, NULL, 0);
    if (stream < 0) {
        set_err(err_buf, err_buf_len, "no matching stream");
        remu_libav_demux_close(session);
        return NULL;
    }
    session->stream_index = stream;
    if (open_decoder_for_stream(session->fmt, stream, &session->codec, err_buf, err_buf_len) < 0) {
        remu_libav_demux_close(session);
        return NULL;
    }
    if (!audio) {
        session->width = session->codec->width;
        session->height = session->codec->height;
        if (session->width <= 0 || session->height <= 0) {
            set_err(err_buf, err_buf_len, "invalid video dimensions");
            remu_libav_demux_close(session);
            return NULL;
        }
    }
    session->frame = av_frame_alloc();
    session->packet = av_packet_alloc();
    if (!session->frame || !session->packet) {
        set_err(err_buf, err_buf_len, "frame/packet alloc failed");
        remu_libav_demux_close(session);
        return NULL;
    }
    if (seek_seconds > 0.0) {
        int64_t ts = (int64_t)(seek_seconds / av_q2d(session->fmt->streams[stream]->time_base));
        if (av_seek_frame(session->fmt, stream, ts, AVSEEK_FLAG_BACKWARD) >= 0) {
            avcodec_flush_buffers(session->codec);
        }
    }
    return session;
}

int remu_libav_demux_read_frame(RemuLibavDemux *session, uint8_t *out, size_t out_len) {
    if (!session || !out) {
        return -1;
    }
    if (session->audio) {
        return write_audio_grain(session, out, out_len);
    }
    for (;;) {
        if (session->eof) {
            /* Drain decoder. */
            avcodec_send_packet(session->codec, NULL);
            int got = avcodec_receive_frame(session->codec, session->frame);
            if (got == 0) {
                return write_video_frame(session, out, out_len);
            }
            return 1;
        }
        int ret = av_read_frame(session->fmt, session->packet);
        if (ret < 0) {
            session->eof = 1;
            continue;
        }
        if (session->packet->stream_index != session->stream_index) {
            av_packet_unref(session->packet);
            continue;
        }
        if (avcodec_send_packet(session->codec, session->packet) < 0) {
            av_packet_unref(session->packet);
            continue;
        }
        av_packet_unref(session->packet);
        if (avcodec_receive_frame(session->codec, session->frame) == 0) {
            return write_video_frame(session, out, out_len);
        }
    }
}

void remu_libav_demux_close(RemuLibavDemux *session) {
    if (!session) {
        return;
    }
    if (session->sws) {
        sws_freeContext(session->sws);
    }
    if (session->swr) {
        swr_free(&session->swr);
    }
    if (session->nv12) {
        av_frame_free(&session->nv12);
    }
    if (session->frame) {
        av_frame_free(&session->frame);
    }
    if (session->packet) {
        av_packet_free(&session->packet);
    }
    if (session->codec) {
        avcodec_free_context(&session->codec);
    }
    if (session->fmt) {
        avformat_close_input(&session->fmt);
    }
    free(session->audio_staging);
    free(session);
}

RemuLibavCodec *remu_libav_codec_open(uint32_t codec_id, char *err_buf, size_t err_buf_len) {
    enum AVCodecID id;
    switch (codec_id) {
        case 1: /* SCE_VIDEODEC2_CODEC_TYPE_AVC */
            id = AV_CODEC_ID_H264;
            break;
        case 974921: /* SCE_VIDEODEC2_CODEC_TYPE_HEVC */
            id = AV_CODEC_ID_HEVC;
            break;
        default:
            set_err(err_buf, err_buf_len, "unsupported videodec2 codecType");
            return NULL;
    }
    const AVCodec *decoder = avcodec_find_decoder(id);
    if (!decoder) {
        set_err(err_buf, err_buf_len, "libavcodec missing decoder");
        return NULL;
    }
    RemuLibavCodec *codec = (RemuLibavCodec *)calloc(1, sizeof(*codec));
    if (!codec) {
        set_err(err_buf, err_buf_len, "oom");
        return NULL;
    }
    codec->codec = avcodec_alloc_context3(decoder);
    codec->frame = av_frame_alloc();
    codec->packet = av_packet_alloc();
    if (!codec->codec || !codec->frame || !codec->packet) {
        set_err(err_buf, err_buf_len, "codec alloc failed");
        remu_libav_codec_close(codec);
        return NULL;
    }
    if (avcodec_open2(codec->codec, decoder, NULL) < 0) {
        set_err(err_buf, err_buf_len, "avcodec_open2 failed");
        remu_libav_codec_close(codec);
        return NULL;
    }
    return codec;
}

int remu_libav_codec_send(
    RemuLibavCodec *codec,
    const uint8_t *au,
    size_t au_size,
    int64_t pts) {
    if (!codec) {
        return -1;
    }
    if (!au || au_size == 0) {
        return avcodec_send_packet(codec->codec, NULL) < 0 ? -1 : 0;
    }
    if (av_new_packet(codec->packet, (int)au_size) < 0) {
        return -1;
    }
    memcpy(codec->packet->data, au, au_size);
    codec->packet->pts = pts;
    int ret = avcodec_send_packet(codec->codec, codec->packet);
    av_packet_unref(codec->packet);
    return ret < 0 ? -1 : 0;
}

static int codec_ensure_nv12(RemuLibavCodec *codec, int width, int height) {
    if (codec->sws && codec->nv12 && codec->width == width && codec->height == height) {
        return 0;
    }
    if (codec->sws) {
        sws_freeContext(codec->sws);
        codec->sws = NULL;
    }
    if (codec->nv12) {
        av_frame_free(&codec->nv12);
    }
    codec->nv12 = av_frame_alloc();
    if (!codec->nv12) {
        return -1;
    }
    codec->nv12->format = AV_PIX_FMT_NV12;
    codec->nv12->width = width;
    codec->nv12->height = height;
    if (av_frame_get_buffer(codec->nv12, 32) < 0) {
        return -1;
    }
    codec->sws = sws_getContext(
        width,
        height,
        codec->codec->pix_fmt,
        width,
        height,
        AV_PIX_FMT_NV12,
        SWS_BILINEAR,
        NULL,
        NULL,
        NULL);
    codec->width = width;
    codec->height = height;
    return codec->sws ? 0 : -1;
}

int remu_libav_codec_receive_nv12(
    RemuLibavCodec *codec,
    uint8_t *out,
    size_t out_len,
    int32_t *width_out,
    int32_t *height_out,
    int32_t *pitch_out) {
    if (!codec || !out || !width_out || !height_out || !pitch_out) {
        return -1;
    }
    int ret = avcodec_receive_frame(codec->codec, codec->frame);
    if (ret == AVERROR(EAGAIN)) {
        return 1;
    }
    if (ret == AVERROR_EOF) {
        return 1;
    }
    if (ret < 0) {
        return -1;
    }
    int width = codec->frame->width;
    int height = codec->frame->height;
    AVFrame *src = codec->frame;
    if (codec->frame->format != AV_PIX_FMT_NV12) {
        if (codec_ensure_nv12(codec, width, height) < 0) {
            return -1;
        }
        if (sws_scale(
                codec->sws,
                (const uint8_t *const *)codec->frame->data,
                codec->frame->linesize,
                0,
                height,
                codec->nv12->data,
                codec->nv12->linesize)
            <= 0) {
            return -1;
        }
        src = codec->nv12;
    }
    if (copy_nv12_from_frame(src, out, out_len, width, height) < 0) {
        return -1;
    }
    *width_out = width;
    *height_out = height;
    *pitch_out = width;
    return 0;
}

int remu_libav_codec_flush(RemuLibavCodec *codec) {
    if (!codec) {
        return -1;
    }
    avcodec_flush_buffers(codec->codec);
    return remu_libav_codec_send(codec, NULL, 0, 0);
}

void remu_libav_codec_close(RemuLibavCodec *codec) {
    if (!codec) {
        return;
    }
    if (codec->sws) {
        sws_freeContext(codec->sws);
    }
    if (codec->nv12) {
        av_frame_free(&codec->nv12);
    }
    if (codec->frame) {
        av_frame_free(&codec->frame);
    }
    if (codec->packet) {
        av_packet_free(&codec->packet);
    }
    if (codec->codec) {
        avcodec_free_context(&codec->codec);
    }
    free(codec);
}

#endif /* REMU_LIBAV_STUB */
