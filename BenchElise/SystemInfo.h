//-----------------------------------------------------------------------------
//								SystemInfo.h
//								============
//
//-----------------------------------------------------------------------------

#ifndef __SYSTEMINFO_H
#define __SYSTEMINFO_H

#include "StdAfx.h"
//#include "shlobj.h"
#include <string>

class CSystemInfo {
public:
	double GetDiskFreeSpace(CString path);

	CString BrowseForFolder(LPCSTR lpszTitle, UINT nFlags = 0x0040);
	bool FindFile(const char* filename);
	bool FindFolder(std::string folder);
	bool CopyDirContent(const char* src, const char* dst);
	bool GetFileSize(const char* src, double &size);
};

#endif //__SYSTEMINFO_H
