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

inline void inverseByteOrder( U_INT1 *p ){}
inline void inverseByteOrder( INT1 *p ){}
inline void inverseByteOrder( U_INT2 *p ){ byte_inv_2(p); }
inline void inverseByteOrder( INT2 *p ){ byte_inv_2(p); }
//inline void inverseByteOrder( U_INT4 *p ){ byte_inv_4(p); }
inline void inverseByteOrder( INT4 *p ){ byte_inv_4(p); }
//inline void inverseByteOrder( U_INT8 *p ){ byte_inv_8(p); }
//inline void inverseByteOrder( INT8 *p ){ byte_inv_8(p); }
inline void inverseByteOrder( REAL4 *p ){ byte_inv_4(p); }
inline void inverseByteOrder( REAL8 *p ){ byte_inv_8(p); }
inline void inverseByteOrder( REAL16 *p ){ byte_inv_16(p); }

