#include "MMVII_util.h"
#include "MMVII_Image2D.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Ptxd.h"
#include "cGdalApi.h"



#ifdef MMVII_KEEP_MMV1_IMAGE
#define MMVII_KEEP_LIBRARY_MMV1 true
# include "V1VII.h"
#endif


using namespace MMVII;

#ifdef MMVII_KEEP_MMV1_IMAGE
bool mmvii_use_mmv1_image=false;
extern std::string MM3DFixeByMMVII; // Declared in MMV1 for its own stuff
#endif


namespace MMVII
{


#ifdef MMVII_KEEP_MMV1_IMAGE
static GenIm::type_el ToMMV1(eTyNums aV2)
{
 // StdOut() << "jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj\n";
    switch (aV2)
    {
    case eTyNums::eTN_INT1 : return GenIm::int1  ;
    case eTyNums::eTN_INT2 : return GenIm::int2  ;
    case eTyNums::eTN_INT4 : return GenIm::int4  ;
    case eTyNums::eTN_U_INT1 : return GenIm::u_int1  ;
    case eTyNums::eTN_U_INT2 : return GenIm::u_int2  ;
    case eTyNums::eTN_U_INT4 : return GenIm::u_int4  ;
    case eTyNums::eTN_REAL4 : return GenIm::real4  ;
    case eTyNums::eTN_REAL8 : return GenIm::real8  ;
    default: ;
    }
    MMVII_INTERNAL_ERROR("GenIm::type_el ToMMV1(eTyNums)");
    return GenIm::int1;
}

static eTyNums ToMMVII( GenIm::type_el aV1 )
{
    switch (aV1)
    {
    case  GenIm::int1 :  return eTyNums::eTN_INT1 ;
    case  GenIm::int2 :  return eTyNums::eTN_INT2 ;
    case  GenIm::int4 :  return eTyNums::eTN_INT4 ;

    case  GenIm::u_int1 :  return eTyNums::eTN_U_INT1 ;
    case  GenIm::u_int2 :  return eTyNums::eTN_U_INT2 ;
    case  GenIm::u_int4 :  return eTyNums::eTN_U_INT4 ;

    case  GenIm::real4 :  return eTyNums::eTN_REAL4 ;
    case  GenIm::real8 :  return eTyNums::eTN_REAL8 ;

    default: ;
    }
    return eTyNums::eTN_UnKnown ;
}


static void Init_mm3d_In_MMVII()
{
    //
    static bool First= true;
    if (! First) return;
    First = false;

    // Compute mm3d location from relative position to MMVII
    // static std::string CA0 =  DirBin2007 + "../../bin/mm3d";
    char * A0= const_cast<char *>(cMMVII_Appli::MMV1Bin().c_str());
    MM3DFixeByMMVII = cMMVII_Appli::MMV1Bin();
    MMD_InitArgcArgv(1,&A0);
}
#endif


/* =========================== */
/*       cDataFileIm2D         */
/* =========================== */

cDataFileIm2D::cDataFileIm2D(const std::string & aName, eTyNums aType, const cPt2di & aSz, int aNbChannel, const tOptions& aOptions, eForceGray isFG, eCreationState aCreationState) :
    cPixBox<2> (cPt2di(0,0),aSz),
    mName       (aName),
    mType       (aType),
    mNbChannel  (aNbChannel),
    mForceGray  (isFG),
    mCreateOptions  (aOptions),
    mExifState      (eExifState::NotRead),
    mCreationState  (aCreationState)
{
}


cDataFileIm2D cDataFileIm2D::Empty()
{
    return cDataFileIm2D( MMVII_NONE, eTyNums::eNbVals, cPt2di(1,1), -1,{},eForceGray::No, eCreationState::Created);
}

bool cDataFileIm2D::IsEmpty() const
{
    return mNbChannel<=0;
}


void cDataFileIm2D::AssertNotEmpty() const
{
    MMVII_INTERNAL_ASSERT_strong((!IsEmpty()),"cDataFileIm2D was not initialized");
}


// Create a cDataFileIm2D on an existing image
cDataFileIm2D cDataFileIm2D::Create(const std::string & aName,eForceGray isFG)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    if (mmvii_use_mmv1_image) {
        // required because with jpg/raw mm1 may call itself, need some special stuff
        // as standar mmv1 by analyse of arg/argv would not work
        Init_mm3d_In_MMVII();

        bool aForce8B = false;
        std::string aNameTif = NameFileStd(aName,-1,!aForce8B ,true,true);
        Tiff_Im aTF = Tiff_Im::StdConvGen(aNameTif.c_str(),-1,!aForce8B ,true);

        return cDataFileIm2D(aName,ToMMVII(aTF.type_el()),ToMMVII(aTF.sz()), aTF.nb_chan(), {}, isFG, eCreationState::Created);
    } else {
#endif
        cPt2di aSz;
        int aNbChannel;
        eTyNums aType;
        cGdalApi::GetFileInfo(aName, aType, aSz, aNbChannel);
        return cDataFileIm2D(aName, aType, aSz, aNbChannel, {}, isFG, eCreationState::Created);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}

cDataFileIm2D  cDataFileIm2D::Create(const std::string & aName,eTyNums aType,const cPt2di & aSz,const tOptions& aOptions, int aNbChan)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    if (mmvii_use_mmv1_image) {
        Tiff_Im::PH_INTER_TYPE aPIT = Tiff_Im::BlackIsZero;
        if (aNbChan==1)
            aPIT = Tiff_Im::BlackIsZero;
        else if (aNbChan==3)
            aPIT = Tiff_Im::RGB;
        else
        {
            MMVII_INTERNAL_ASSERT_strong(false,"Incoherent channel number");
        }

        bool IsModified;
        Tiff_Im::CreateIfNeeded
            (
                IsModified,
                aName,
                ToMMV1(aSz),
                ToMMV1(aType),
                Tiff_Im::No_Compr,
                aPIT
                );
        return Create(aName,eForceGray::No);
    } else {
#endif
        if (aNbChan!=1 && aNbChan!=3)
        {
            MMVII_INTERNAL_ASSERT_strong(false,"Incoherent channel number");
        }
        // eCreationState::AtFirstWrite: If needed, file will be created with a null image and new state will be Created or CreatedNoUpdate
        cDataFileIm2D aDataFileIm2D(aName, aType, aSz, aNbChan, aOptions, eForceGray::No, eCreationState::AtFirstWrite);
        cGdalApi::CreateFileIfNeeded(aDataFileIm2D);
        return aDataFileIm2D;
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}

cDataFileIm2D  cDataFileIm2D::Create(const std::string & aName,eTyNums aType,const cPt2di & aSz, int aNbChan)
{
    return Create(aName,aType,aSz,{},aNbChan);
}

cDataFileIm2D  cDataFileIm2D::CreateOnWrite(const std::string & aName,eTyNums aType,const cPt2di & aSz, const tOptions& aOptions, int aNbChan)
{
    if (aNbChan!=1 && aNbChan!=3)
    {
        MMVII_INTERNAL_ASSERT_strong(false,"Incoherent channel number");
    }
    cGdalApi::InitGDAL();
    return cDataFileIm2D(aName, aType, aSz, aNbChan, aOptions, eForceGray::No, eCreationState::AtFirstWrite);
}

cDataFileIm2D  cDataFileIm2D::CreateOnWrite(const std::string & aName,eTyNums aType,const cPt2di & aSz, int aNbChan)
{
    return CreateOnWrite(aName,aType,aSz,{},aNbChan);
}


cDataFileIm2D::~cDataFileIm2D()
{
}

const cPt2di &  cDataFileIm2D::Sz() const  {return  cPixBox<2>::Sz();}
const std::string &  cDataFileIm2D::Name() const { return mName; }
const int  & cDataFileIm2D::NbChannel ()  const { return mNbChannel; }
const eTyNums &   cDataFileIm2D::Type ()  const {return mType;}
bool  cDataFileIm2D::IsCreateAtFirstWrite() const  {return  mCreationState == eCreationState::AtFirstWrite;}
bool  cDataFileIm2D::IsCreatedNoUpdate() const  {return  mCreationState == eCreationState::CreatedNoUpdate;}
const cDataFileIm2D::tOptions& cDataFileIm2D::CreateOptions() const { return mCreateOptions; }

void cDataFileIm2D::SetCreated() const
{
    mCreationState = eCreationState::Created;
}

void cDataFileIm2D::SetCreatedNoUpdate() const
{
    mCreationState = eCreationState::CreatedNoUpdate;
}

const cExifData& cDataFileIm2D::ExifDataAll(bool SVP) const
{
    if (mExifState != eExifState::AllTagsRead)
    {
        mExifData.FromFile(mName,SVP);
        mExifState = eExifState::AllTagsRead;

    }
    return mExifData;
}

const cExifData& cDataFileIm2D::ExifDataMain(bool SVP) const
{
    if (mExifState != eExifState::MainTagsRead && mExifState != eExifState::AllTagsRead)
    {
        mExifData.FromFileMainOnly(mName,SVP);
        mExifState = eExifState::MainTagsRead;
    }
    return mExifData;
}

std::vector<std::string> cDataFileIm2D::ExifStrings(bool SVP) const
{
    return cExifData::StringListFromFile(mName,SVP);
}

std::map<std::string, std::vector<std::string>> cDataFileIm2D::AllMetadata(bool SVP) const
{
    return cExifData::AllMetadataFromFile(mName,SVP);
}



bool cDataFileIm2D::IsPostFixNameImage(const std::string & aPost)
{
    return cGdalApi::IsPostFixNameImage(aPost);
}

bool cDataFileIm2D::IsNameWith_PostFixImage(const std::string & aName)
{
    return IsPostFixNameImage(LastPostfix(aName));
}



#ifdef MMVII_KEEP_MMV1_IMAGE

template <class Type> void cMMV1_Conv<Type>::ReadWrite
    (
        bool ReadMode,
        const std::vector<const tImMMVII *> & aVecImV2,
        const cDataFileIm2D & aDF,
        const cPt2di & aP0File,
        double aDyn,
        const cRect2& aR2Init
        )
{

    // StdOut() <<  "aP0File,aP0File, " << aP0File << "\n";
    Init_mm3d_In_MMVII();
    // C'est une image en originie (0,0) necessairement en MMV1
    const tImMMVII & aImV2 = *(aVecImV2.at(0));
    Fonc_Num aFoncImV1 = ImToMMV1(aImV2).in();
    Output   aOutImV1  = ImToMMV1(aImV2).out();
    for (int aKIm=1 ; aKIm<int(aVecImV2.size()) ; aKIm++)
    {
        MMVII_INTERNAL_ASSERT_strong(aImV2.Sz()==aVecImV2.at(aKIm)->Sz(),"Diff Sz in ReadWrite");
        MMVII_INTERNAL_ASSERT_strong(aImV2.P0()==aVecImV2.at(aKIm)->P0(),"Diff P0 in ReadWrite");
        aFoncImV1 = Virgule(aFoncImV1,ImToMMV1(*aVecImV2.at(aKIm)).in());
        aOutImV1  = Virgule( aOutImV1,ImToMMV1(*aVecImV2.at(aKIm)).out());
    }
    cRect2 aRectFullIm (cPt2di(0,0),aImV2.Sz());

    // Rectangle image / a un origine (0,0)
    cRect2 aRectIm =  (aR2Init== cRect2::TheEmptyBox)           ?  // Val par def
                         aRectFullIm                           :  // Rectangle en 00
                         aR2Init.Translate(-aImV2.P0())   ;  // Convention aR2Init tient compte de P0

    // It's a bit strange but in fact P0File en aImV2.P0() are redundant, so if both are used
    // it seems normal to add them

    Pt2di aTrans  = ToMMV1(aImV2.P0() + aP0File);
    Pt2di aP0Im = ToMMV1(aRectIm.P0());
    Pt2di aP1Im = ToMMV1(aRectIm.P1());

    cRect2 aRUsed(ToMMVII(aP0Im+aTrans),ToMMVII(aP1Im+aTrans));
    if (true)
    {
        MMVII_INTERNAL_ASSERT_strong(aRUsed.IncludedIn(aDF), "Read/write out of file");
        MMVII_INTERNAL_ASSERT_strong(aRectIm.IncludedIn(aRectFullIm), "Read/write out of Im");
    }

    Tiff_Im aTF=Tiff_Im::StdConvGen(aDF.Name(),-1,true);

    if (ReadMode)
    {
        Symb_FNum  aFIn = aTF.in();
        // If input is multi-channel and out single, we compute the average of all channel
        if ((aFIn.dimf_out()>1) && (aVecImV2.size()==1))
        {
            // if we have more than 3 channel, the 4th is generally an alpha channel, so dont want to use it
            // btw it a bit basic, and certainly a more sophisticated rule will have to be used (as weighting of diff channels)
            int aNbCh = std::min(3,aFIn.dimf_out());

            Fonc_Num aNewF = aFIn.kth_proj(0);
            for (int aKF=1 ; aKF< aNbCh ; aKF++)
                aNewF = aNewF + aFIn.kth_proj(aKF);

            aFIn = aNewF / aNbCh;
        }

        ELISE_COPY
            (
                rectangle(aP0Im,aP1Im),
                trans(El_CTypeTraits<Type>::TronqueF(aFIn*aDyn),aTrans),
                aOutImV1
                );
    }
    else
    {
        ELISE_COPY
            (
                rectangle(aP0Im+aTrans,aP1Im+aTrans),
                trans(Tronque(aTF.type_el(),aFoncImV1*aDyn),-aTrans),
                aTF.out()
                );
    }
}

template <class Type> void cMMV1_Conv<Type>::ReadWrite
    (
        bool ReadMode,
        const tImMMVII &aImV2,
        const cDataFileIm2D & aDF,
        const cPt2di & aP0File,
        double aDyn,
        const cRect2& aR2Init
        )
{
    std::vector<const tImMMVII *> aVIms({&aImV2});
    ReadWrite(ReadMode,aVIms,aDF,aP0File,aDyn,aR2Init);
}
template <class Type> void cMMV1_Conv<Type>::ReadWrite
    (
        bool ReadMode,
        const tImMMVII &aImV2R,
        const tImMMVII &aImV2G,
        const tImMMVII &aImV2B,
        const cDataFileIm2D & aDF,
        const cPt2di & aP0File,
        double aDyn,
        const cRect2& aR2Init
        )
{

    std::vector<const tImMMVII *> aVIms({&aImV2R,&aImV2G,&aImV2B});
    ReadWrite(ReadMode,aVIms,aDF,aP0File,aDyn,aR2Init);
}


template <> void cMMV1_Conv<tREAL16>::ReadWrite
    (bool,const tImMMVII &,const cDataFileIm2D &,const cPt2di &,double,const cRect2& )
{
    MMVII_INTERNAL_ASSERT_strong(false,"No ReadWrite of 16-Byte float");
}
template <> void cMMV1_Conv<tREAL16>::ReadWrite
    (bool,const tImMMVII &,const tImMMVII &,const tImMMVII &,const cDataFileIm2D &,const cPt2di &,double,const cRect2& )
{
    MMVII_INTERNAL_ASSERT_strong(false,"No ReadWrite of 16-Byte float");
}
template <> void cMMV1_Conv<tU_INT4>::ReadWrite
    (bool,const tImMMVII &,const tImMMVII &,const tImMMVII &,const cDataFileIm2D &,const cPt2di &,double,const cRect2& )
{
    MMVII_INTERNAL_ASSERT_strong(false,"No ReadWrite of 16-Byte float");
}
#endif

template <class Type>  void  cDataIm2D<Type>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    if (mmvii_use_mmv1_image) {
        cMMV1_Conv<Type>::ReadWrite(true,*this,aFile,aP0,aDyn,aR2);
    } else {
#endif
        cGdalApi::ReadWrite(cGdalApi::IoMode::Read, *this, aFile, aP0, aDyn, aR2);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}

template <class Type>  void  cDataIm2D<Type>::Read(const cDataFileIm2D & aFile,tIm &aImG,tIm &aImB,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    if (mmvii_use_mmv1_image) {
        cMMV1_Conv<Type>::ReadWrite(true,*this,aImG,aImB,aFile,aP0,aDyn,aR2);
    } else {
#endif
        cGdalApi::ReadWrite(cGdalApi::IoMode::Read, *this, aImG, aImB, aFile, aP0, aDyn, aR2);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}


template <class Type>  void  cDataIm2D<Type>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    if (mmvii_use_mmv1_image) {
        cMMV1_Conv<Type>::ReadWrite(false,*this,aFile,aP0,aDyn,aR2);
    } else {
#endif
        cGdalApi::ReadWrite(cGdalApi::IoMode::Write, *this, aFile, aP0, aDyn, aR2);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}

template <class Type>  void  cDataIm2D<Type>::Write(const cDataFileIm2D & aFile,const tIm &aImG,const tIm &aImB,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    if (mmvii_use_mmv1_image) {
        cMMV1_Conv<Type>::ReadWrite(false,*this,aImG,aImB,aFile,aP0,aDyn,aR2);
    } else {
#endif
        cGdalApi::ReadWrite(cGdalApi::IoMode::Write, *this, aImG, aImB, aFile, aP0, aDyn, aR2);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}

#ifdef MMVII_KEEP_MMV1_IMAGE
//  It's difficult to read unsigned int4 with micmac V1, wait for final implementation
template <>  void  cDataIm2D<tU_INT4>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
    if (mmvii_use_mmv1_image) {
        MMVII_INTERNAL_ASSERT_strong(false,"No read for unsigned int4 now");
    } else {
        cGdalApi::ReadWrite(cGdalApi::IoMode::Read, *this, aFile, aP0, aDyn, aR2);
    }
}

template <>  void  cDataIm2D<tU_INT4>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
    if (mmvii_use_mmv1_image) {
        MMVII_INTERNAL_ASSERT_strong(false,"No write for unsigned int4 now");
    } else {
        cGdalApi::ReadWrite(cGdalApi::IoMode::Write, *this, aFile, aP0, aDyn, aR2);
    }
}

template <>  void  cDataIm2D<tU_INT4>::Write(const cDataFileIm2D & aFile,const tIm& aImG,const tIm& aImB,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
    if (mmvii_use_mmv1_image) {
        MMVII_INTERNAL_ASSERT_strong(false,"No write for unsigned int4 now");
    } else {
        cGdalApi::ReadWrite(cGdalApi::IoMode::Write, *this, aImG, aImB, aFile, aP0, aDyn, aR2);
    }
}
#endif


template <class Type>  void  cIm2D<Type>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
    DIm().Read(aFile,aP0,aDyn,aR2);
}
template <class Type>  void  cIm2D<Type>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
    DIm().Write(aFile,aP0,aDyn,aR2);
}



double DifAbsInVal(const std::string & aN1,const std::string & aN2,double aDef)
{
    auto  aIm1 = cIm2D<tREAL8>::FromFile(aN1);
    auto  aIm2 = cIm2D<tREAL8>::FromFile(aN2);
    double aSom = 0;
    
    
//    ELISE_COPY(aF1.all_pts(),Abs(aF1.in()-aF2.in()),sigma(aSom));
    
    if (aIm1.DIm().Sz()!=aIm2.DIm().Sz())
    {
        MMVII_INTERNAL_ASSERT_always(aDef!=0.0,"Diff sz and bad def in DifAbsInVal");
        return aDef;
    }

    for (const auto& aPt : aIm1.DIm())
    {
        aSom += std::fabs(aIm1.DIm().GetV(aPt) - aIm2.DIm().GetV(aPt));
    }
    return aSom;
}


//  INSTANTIATION

#define MACRO_INSTANTIATE_READ_FILE(Type)\
template class cDataIm2D<Type>;\
template class cIm2D<Type>;

MACRO_INSTANTIATE_READ_FILE(tINT1)
MACRO_INSTANTIATE_READ_FILE(tINT2)
MACRO_INSTANTIATE_READ_FILE(tINT4)
// MACRO_INSTANTIATE_READ_FILE(tINT8)
MACRO_INSTANTIATE_READ_FILE(tU_INT1)
MACRO_INSTANTIATE_READ_FILE(tU_INT2)
MACRO_INSTANTIATE_READ_FILE(tU_INT4)
MACRO_INSTANTIATE_READ_FILE(tREAL4)
MACRO_INSTANTIATE_READ_FILE(tREAL8)
MACRO_INSTANTIATE_READ_FILE(tREAL16)
};
