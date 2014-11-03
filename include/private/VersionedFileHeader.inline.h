// this file is supposed to be included only in VersionedHeader.h

//--------------------------------------------
// class VersionedFileHeader
//--------------------------------------------

VersionedFileHeader::VersionedFileHeader(){}

bool VersionedFileHeader::isMSBF() const { return (m_isMSBF==1); }

uint32_t VersionedFileHeader::version() const { return m_version; }

uint32_t VersionedFileHeader::magicNumber() const { return m_magicNumber; }


//--------------------------------------------
// related functions
//--------------------------------------------

inline int nbVersionedTypes(){ return (int)VFH_Unknown; }
