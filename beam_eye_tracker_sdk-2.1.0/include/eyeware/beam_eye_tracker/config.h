/**
 * Copyright (C) 2025 Eyeware Tech SA
 *
 * All rights reserved
 */

#ifndef _EYEWARE_BEAM_EYE_TRACKER_CONFIG_H_
#define _EYEWARE_BEAM_EYE_TRACKER_CONFIG_H_

#if !defined(EW_BET_API_EXPORT_DECL)
#if defined(EW_BET_STATIC_LIB)
#define EW_BET_API_EXPORT_DECL
#elif defined(WIN32) || defined(_WIN32)
#define EW_BET_API_EXPORT_DECL __declspec(dllexport)
#else
#define EW_BET_API_EXPORT_DECL __attribute__((visibility("default")))
#endif
#endif

#if !defined(EW_BET_API_IMPORT_DECL)
#if defined(EW_BET_STATIC_LIB)
#define EW_BET_API_IMPORT_DECL
#elif defined(WIN32) || defined(_WIN32)
#define EW_BET_API_IMPORT_DECL __declspec(dllimport)
#else
#define EW_BET_API_IMPORT_DECL
#endif
#endif

#if !defined(EW_BET_API_DECL)
#if defined(EW_BET_API_EXPORTING)
#define EW_BET_API_DECL EW_BET_API_EXPORT_DECL
#else
#define EW_BET_API_DECL EW_BET_API_IMPORT_DECL
#endif
#endif

#endif // _EYEWARE_BEAM_EYE_TRACKER_CONFIG_H_
