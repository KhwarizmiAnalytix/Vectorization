/*
 * XSigma Vectorization library — DLL export/import (same pattern as Parallel / Logging).
 */
#pragma once

#if defined(VECTORIZATION_STATIC_DEFINE)
#define VECTORIZATION_API

#elif defined(VECTORIZATION_SHARED_DEFINE)
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef VECTORIZATION_BUILDING_DLL
#define VECTORIZATION_API __declspec(dllexport)
#else
#define VECTORIZATION_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define VECTORIZATION_API __attribute__((visibility("default")))
#else
#define VECTORIZATION_API
#endif

#else
#define VECTORIZATION_API
#endif
