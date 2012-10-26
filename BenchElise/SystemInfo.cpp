//-----------------------------------------------------------------------------
//								SystemInfo.cpp
//								==============
//
//-----------------------------------------------------------------------------

#include "SystemInfo.h"
#include <fstream>


//-----------------------------------------------------------------------------
// Donne l'espace disque disponible dans un repertoire
//-----------------------------------------------------------------------------
double CSystemInfo::GetDiskFreeSpace(CString path)
{
	typedef BOOL (CALLBACK* LPFNDLLFUNC1)(LPCTSTR, PULARGE_INTEGER,
				  PULARGE_INTEGER , PULARGE_INTEGER );

	HINSTANCE hDLL;               // Handle sur la DLL
	LPFNDLLFUNC1 lpfnDllFunc1;    // Pointeur de fonction
	double diskSpace = 0.0;

	hDLL = LoadLibrary("Kernel32.dll");
	if (hDLL == NULL)
		return 0;

	lpfnDllFunc1 = (LPFNDLLFUNC1)GetProcAddress(hDLL,"GetDiskFreeSpaceExA");
	if (!lpfnDllFunc1)
	{
		DWORD SectorsPerCluster, BytesPerSector, NumberOfFreeClusters, TotalNumberOfClusters;
		SetCurrentDirectory(LPCTSTR(path));
		if (::GetDiskFreeSpace(NULL, &SectorsPerCluster, &BytesPerSector,
			&NumberOfFreeClusters,&TotalNumberOfClusters) == TRUE) {
			diskSpace = (double)(NumberOfFreeClusters * BytesPerSector * SectorsPerCluster);
			diskSpace /= 1048576.0;
		}
	} else { //La fonction GetDiskFreeSpaceEx est connue
		ULARGE_INTEGER FreeBytesAvailableToCaller,TotalNumberOfBytes,TotalNumberOfFreeBytes;
		if (lpfnDllFunc1(LPCTSTR(path),&FreeBytesAvailableToCaller, 
						&TotalNumberOfBytes, &TotalNumberOfFreeBytes ) == TRUE) {
			if (FreeBytesAvailableToCaller.HighPart > 0)
				diskSpace = (double)FreeBytesAvailableToCaller.HighPart * 4096.0 + 
				(double)FreeBytesAvailableToCaller.LowPart / 1048576.0;
			else
				diskSpace = (double)FreeBytesAvailableToCaller.LowPart / 1048576.0;
		}
	}
	FreeLibrary(hDLL);       
	return diskSpace;
}

//-----------------------------------------------------------------------------
// Permet de chercher un repertoire
//-----------------------------------------------------------------------------
CString CSystemInfo::BrowseForFolder(LPCSTR lpszTitle, UINT nFlags)
{

  CString strResult = "";
  
  LPMALLOC lpMalloc;  // pointer to IMalloc

  if (::SHGetMalloc(&lpMalloc) != NOERROR)
     return strResult; // failed to get allocator

  char szDisplayName[_MAX_PATH];
  char szBuffer[_MAX_PATH];

  BROWSEINFO browseInfo;
  browseInfo.hwndOwner = AfxGetMainWnd()->GetSafeHwnd();
  browseInfo.pidlRoot = NULL; 
  browseInfo.pszDisplayName = szDisplayName;
  browseInfo.lpszTitle = lpszTitle;   
  browseInfo.ulFlags = nFlags;   
  browseInfo.lpfn = NULL;      
  browseInfo.lParam = 0;  

  LPITEMIDLIST lpItemIDList = ::SHBrowseForFolder(&browseInfo);
	if (lpItemIDList == NULL) {
		lpMalloc->Free(lpItemIDList);
		lpMalloc->Release();      
		return strResult;
	}

	if (::SHGetPathFromIDList(lpItemIDList, szBuffer)){
    if (szBuffer[0] == '\0')
			return strResult;
    strResult = szBuffer;
    return strResult;
  } else 
    return strResult; // strResult is empty 
  
	lpMalloc->Free(lpItemIDList);
  lpMalloc->Release();      

	return strResult;
}

//-----------------------------------------------------------------------------
// Permet de tester l'existence d'un fichier
//-----------------------------------------------------------------------------
bool CSystemInfo::FindFile(const char* filename)
{
	WIN32_FIND_DATA findData;
	HANDLE file = FindFirstFile(filename, &findData);
	if (file == INVALID_HANDLE_VALUE)
		return false;	// Le fichier n'existe pas
	FindClose(file);
	return true;
}
//-----------------------------------------------------------------------------
// Permet de tester l'existence d'un répertoire
//-----------------------------------------------------------------------------
bool CSystemInfo::FindFolder(std::string folder)
{
	if (folder.size() == 0)
		return false;
	if (folder.substr(folder.size()-1,1).compare("\\") ==0)
		folder = folder.substr(0,folder.size()-1);

	WIN32_FIND_DATA findData;
	HANDLE file = FindFirstFile(folder.c_str(), &findData);
	if (file == INVALID_HANDLE_VALUE)// Le repertoire n'existe pas
		return false;	
	FindClose(file);

	return true;
}

//-----------------------------------------------------------------------------
// Copie du contenu d'un repertoire dans un autre
//-----------------------------------------------------------------------------
bool CSystemInfo::CopyDirContent(const char* src, const char* dst)
{
	CString strSrc = src;
	CString strDst = dst;
	if (strSrc[strSrc.GetLength()-1] != '\\')
		strSrc += "\\";
	if (strDst[strDst.GetLength()-1] != '\\')
		strDst += "\\";

	CString	oldFile, newFile, strPath = strSrc + "*.*";
		
	HANDLE hFind;
	WIN32_FIND_DATA fd;		
		
  if ((hFind = ::FindFirstFile (LPCTSTR(strPath), &fd)) != INVALID_HANDLE_VALUE) {
		if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			newFile = strDst + fd.cFileName;
			oldFile = strSrc + fd.cFileName;
			::CopyFile(LPCTSTR(oldFile), LPCTSTR(newFile), FALSE);
		} 
	}
	else
		return false;
		
	while (::FindNextFile (hFind, &fd)) {
		if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			newFile = strDst + fd.cFileName;
			oldFile = strSrc + fd.cFileName;
			::CopyFile(LPCTSTR(oldFile), LPCTSTR(newFile), FALSE);
		} 
	}	
	::FindClose(hFind);
	return true;
}


//-----------------------------------------------------------------------------
// Renvoie true si le fichier existe avec sa taille
//-----------------------------------------------------------------------------
bool CSystemInfo::GetFileSize(const char* src, double &size)
{
	size = 0;

	std::ifstream fic(src);
	if (!fic.good())
		return false;
	fic.seekg(0, std::ios::end);
	size = fic.tellg();
	return true;
}