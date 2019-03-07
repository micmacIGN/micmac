/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/


#ifndef _ELISE_XML_GEN_MMBY_P_
#define _ELISE_XML_GEN_MMBY_P_

std::string StdMetaDataFilename(const std::string &aBasename, bool aBinary);

cXmlXifInfo MDT2Xml(const cMetaDataPhoto & aMTD);

void MakeXmlXifInfo(const std::string & aFullPat,cInterfChantierNameManipulateur * aICNM,bool toForce=false);

const std::string  DirFusMMInit();
const std::string  DirFusStatue();
const std::string  PrefDNF() ; //  DownScale_NuageFusion-

const std::string TheDIRMergTiepForEPI();
const std::string TheDIRMergeEPI();
const std::string TheRaffineQuickMac();
const std::string TheRaffineQuickMac(const std::string &);


const std::string & TheNameMatrCorrelDir();
const std::string & TheNameMatrCorrelInv();
const std::string & TheNameMatrCov();
const std::string & TheNameMatrCorrel();
bool IsMatriceExportBundle(const std::string &);

const std::string & TheNameFileVarName();
const std::string & TheNameFileSensibVar();



extern const std::string ExtTxtXml;
extern const std::string ExtBinDmp;
const std::string & ExtXml(bool Bin);

class cImaMM;
class cAppliWithSetImage;
class cAppliMMByPair;

class cAttrSomAWSI
{
    public :
        cAttrSomAWSI();
        cAttrSomAWSI(cImaMM*,int aNumGlob,int aNumAccepted);
        cImaMM* mIma;
        int     mNumGlob;
        int     mNumAccepted;
};

class cAttrArcAWSI
{
    public :
       cAttrArcAWSI(cCpleEpip *);
       cAttrArcAWSI();

       cCpleEpip * mCpleE;
};

typedef  ElSom<cAttrSomAWSI,cAttrArcAWSI>         tSomAWSI;
typedef  ElArc<cAttrSomAWSI,cAttrArcAWSI>         tArcAWSI;
typedef  ElSomIterator<cAttrSomAWSI,cAttrArcAWSI> tItSAWSI;
typedef  ElArcIterator<cAttrSomAWSI,cAttrArcAWSI> tItAAWSI;
typedef  ElGraphe<cAttrSomAWSI,cAttrArcAWSI>      tGrAWSI;

std::string NameImage(tArcAWSI &,bool Im1,bool ByEpi);


std::string PatternOfVois(const tSomAWSI & ,bool IncludeThis) ;


class cImaMM
{
    public :
      cImaMM(const std::string & aName,cAppliWithSetImage &);


    public :
       CamStenope * CamSNN();
       CamStenope * CamSSvp();
       cBasicGeomCap3D *  CamGen();
       Tiff_Im  &   TiffStd();
       Tiff_Im  &   Tiff8BGr();
       Tiff_Im  &   Tiff8BCoul();
       Tiff_Im  &   Tiff16BGr();
    private :
       cBasicGeomCap3D *     mCamGen;
       CamStenope *          mCamS;
    public :
       std::string mNameIm;
       std::string mBande;
       int         mNumInBande;
       Pt3dr        mC3;
       Pt2dr        mC2;
       cAppliWithSetImage &  mAppli;
       Tiff_Im  *            mPtrTiffStd;
       Tiff_Im  *            mPtrTiff8BGr;
       Tiff_Im  *            mPtrTiff8BCoul;
       Tiff_Im  *            mPtrTiff16BGr;

};

inline Pt2dr PtOfSomAWSI    (const tSomAWSI & aS) {return  aS.attr().mIma->mC2;}
inline Pt2dr PtOfSomAWSIPtr (const tSomAWSI * aS) {return PtOfSomAWSI(*aS);}



class cSubGrAWSI : public ElSubGraphe<cAttrSomAWSI,cAttrArcAWSI>
{
    public :
        Pt2dr pt(tSomAWSI & aS) {return PtOfSomAWSI(aS);}
};

class cElemAppliSetFile
{
    public :
       cElemAppliSetFile();
       cElemAppliSetFile(const std::string &);
       void Init(const std::string &);


       std::string mFullName;
       std::string mDir;
       std::string mPat;
       cInterfChantierNameManipulateur * mICNM;
       const cInterfChantierNameManipulateur::tSet * SetIm();
    protected :
       const cInterfChantierNameManipulateur::tSet * mSetIm;
};


std::string PatFileOfImSec(const std::string & anOri);
std::string DirAndPatFileOfImSec(const std::string & aDir,const std::string & anOri);
std::string DirAndPatFileMMByP(const std::string & aDir);



class cAppliWithSetImage
{
   public :
      std::vector<CamStenope*> VCamStenope();
      std::vector<ElCamera*>   VCam();

      cBasicGeomCap3D * CamGenOfName(const std::string & aName);
      CamStenope * CamOfName(const std::string & aName);
      const std::string & Dir() const;
      const std::string & Ori() const;
      bool HasOri() const;
      cInterfChantierNameManipulateur * ICNM() ;
      int  DeZoomOfSize(double ) const;
      void operator()(tSomAWSI*,tSomAWSI*,bool);   // Delaunay call back

    // Remplace la commande argc-argc par N command avec les image indiv, aNumPat est necessaire car peut varier (TestLib ou non)
      // Probably WithDir=true in most case, but for perfect backward compatibility set it to false
      std::list<std::pair<std::string,std::string> > ExpandCommand(int aNumPat,std::string ArgSup,bool Exe=false,bool WithDir=false);

      static const int  TheFlagDev8BGray      = 1;
      static const int  TheFlagDev16BGray     = 2;
      static const int  TheFlagNoOri          = 4;  
      static const int  TheFlagAcceptProblem  = 8;  
      static const int  TheFlagDev8BCoul      = 16;
      static const int  TheFlagDevXml         = 32;
  
      cAppliWithSetImage(int argc,char ** argv,int aFlag,const std::string & aNameCAWSI="");
      std::string PatFileOfImSec() const;
      std::string DirAndPatFileOfImSec() const;
      std::string DirAndPatFileMMByP() const;
      void SuppressSom(tSomAWSI & aSom);

      static const std::string TheMMByPairNameCAWSI;
      static const std::string TheMMByPairNameFiles;
      cElemAppliSetFile & EASF();
      const cElemAppliSetFile & EASF() const;

   protected :

      void SaveCAWSI(const std::string & aName) ;
      bool CAWSI_AcceptIm(const std::string & aName) const;
      bool CAWSI_AcceptCpleIm(const std::string & aN1,const std::string &  aN2) const;
      
      // Si AnalysConexion = false, revoit juste le SET 
      void FilterImageIsolated(bool AnalysConexions=true);
      void Develop(bool EnGray,bool En16B);
      bool MasterSelected(const std::string & aName) const;
      bool MasterSelected(tSomAWSI* aSom) const;
      bool CpleHasMasterSelected(tSomAWSI* aS1,tSomAWSI* aS2) const;



      tSomAWSI * ImOfName(const std::string & aName);
      bool ImIsKnown(const std::string & aName) const;

      void MakeStripStruct(const std::string & aPairByStrip,bool StripFirst);
      void AddDelaunayCple();
      void AddFilePair(const std::string & aFilePair);
      void AddCoupleMMImSec(bool ExeApero,bool SupressImInNoMasq,bool AddCple, const std::string &SetHom, bool ExpTxt=false,bool ExpImSec=true,double aTetaOpt=0.17);
      void AddLinePair(int aDif, bool ExpTxt);


      void DoPyram();

      void VerifAWSI();
      void ComputeStripPair(int);
      void AddPair(tSomAWSI * anI1,tSomAWSI * anI2);
      void AddPair(const std::string & aN1,const std::string & aN2,bool aSVP);

      bool        mSym;
      bool        mShow;
      std::string mPb;
/*
      std::string mFullName;
      std::string mDir;
      std::string mPat;
*/
      bool        mWithOri;
      std::string mOri;
      std::string mKeyOri;
      cElemAppliSetFile mEASF;
      // cInterfChantierNameManipulateur * mICNM;
      // const cInterfChantierNameManipulateur::tSet * mSetIm;

      std::map<std::string,tSomAWSI *> mDicIm;
      tGrAWSI  mGrIm;
      std::vector<tSomAWSI*> mVSoms;
      bool                             mWithCAWSI;
      std::map<std::string,cCWWSImage> mDicWSI;

      cSubGrAWSI   mSubGrAll;
      double       mAverNbPix;



      double       mTetaBande;
      bool         mByEpi;

      int NbAlti() const;
      double AltiMoy() const;
      cSetName *   mSetMasters;
      bool mCalPerIm;
      double mPenPerIm;
      bool mModeHelp;
      std::string  mMasq3D;
      std::vector<std::string>         mVNameFinal;



   private :
      int   mNbAlti;
      double mSomAlti;
      bool   mSupressImInNoMasq;
      const std::vector<std::string> * mSetImNoMasq;
};


template <class eType> std::list<std::string> ListOfVal(eType aValMax,const std::string& ToSub="e") // Exclue
{
    std::list<std::string> aRes;
    for (int aK=0 ; aK<int(aValMax) ; aK++)
    {
        std::string aVal = eToString((eType) aK);
        aRes.push_back(aVal.substr(ToSub.size(),std::string::npos));
    }
    return aRes;
}

void StdCorrecNameHomol(std::string & aNameH,const std::string & aDir);

bool StdCorrecNameOrient(std::string & aNameOri,const std::string & aDir,bool SVP=false);
void   CorrecNameMasq(const std::string & aDir,const std::string & aPat,std::string & aMasq);
cSpecifFormatRaw * GetSFRFromString(const std::string & aNameHdr);

class cPatOfName
{
    public :
       cPatOfName();
       std::string Pattern() const;
       void AddName(const std::string &);
    private :
        std::string mPat;
        int mNb;
};

void DoAllDev(const std::string & aPat);
void GenTFW(const ElAffin2D & anAff,const std::string & aNameTFW);
void GenTFW(const cFileOriMnt & aFOM,const std::string & aName);
double ResolOfAff(const ElAffin2D & anAff);
Box2dr BoxTerOfNu(const cXML_ParamNuage3DMaille & aNu);
double ResolOfNu(const cXML_ParamNuage3DMaille & aNu);


class cChantierAppliWithSetImage;
class cCWWSImage;
const cCWWSImage * GetFromCAWSI(const cChantierAppliWithSetImage & ,const std::string & );

class cResVINM
{
   public :
       cResVINM();

       CamStenope*     mCam1;
       CamStenope*     mCam2;
       cElHomographie  mHom;
       double          mResHom;
   private :
};

typedef std::vector<Pt2df> tVP2f;
typedef const tVP2f   tCVP2f;
typedef std::vector<U_INT1> tVUI1;
typedef const tVUI1 tCVUI1;


class cVirtInterf_NewO_NameManager
{
       public :
           virtual void WriteTriplet(const std::string & aNameFile,tCVP2f &,tCVP2f &,tCVP2f &,tCVUI1 &)=0;
           virtual void WriteCouple(const std::string & aNameFile,tCVP2f &,tCVP2f &,tCVUI1 &) = 0;

           virtual std::string NameRatafiaSom(const std::string & aName,bool Bin) const = 0;
           virtual std::string NameListeCpleOriented(bool Bin) const = 0;
           virtual std::string NameListeCpleConnected(bool Bin) const = 0;

           virtual CamStenope * OutPutCamera(const std::string & aName) const = 0;
           virtual CamStenope * CalibrationCamera(const std::string  & aName) const = 0;
           // for a given image "aName", return the list of images having homolgous data (tieP + orientaion)

           virtual std::list<std::string>  ListeImOrientedWith(const std::string & aName) const = 0;

           virtual std::pair<CamStenope*,CamStenope*> CamOriRel(const std::string &,const std::string &) const =0;
           virtual cResVINM  ResVINM(const std::string &,const std::string &) const =0;

           


           // for a given pair of image, load the tie points (in two vector of point)
           //  !! => they are "photogrametric" tie points, i.e they have been corrected of focal, PP and distorsion
           //  for a given 2d point (U,V)  the (U,V,1) 3d point is a direction in the camera repair
           virtual void LoadHomFloats(std::string,std::string,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,bool SVP=false) = 0;
           virtual void GenLoadHomFloats(const std::string &  aNameH,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,bool SVP)=0;

           virtual bool LoadTriplet(const std::string &,const std::string &,const std::string &,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,std::vector<Pt2df> * aVP3) = 0;


           // for a given pair of image, return the structure containg the orientation
           virtual cXml_Ori2Im GetOri2Im(const std::string & aN1,const std::string & aN2) const = 0;


           static cVirtInterf_NewO_NameManager * StdAlloc(
                                                            const std::string  & aDir,   // Global Dir
                                                            const std::string  & anOri,  // Dir where is stored calibration
                                                            bool  Quick  = true  // Mean that accelarated computation where done
                                                );
           // === surcharge method avant pour adapter avec suffix homologue =======
           static cVirtInterf_NewO_NameManager * StdAlloc(
                                                            const std::string  & aPrefHom,
                                                            const std::string  & aDir,   // Global Dir
                                                            const std::string  & anOri,  // Dir where is stored calibration
                                                            bool  Quick  = true  // Mean that accelarated computation where done
                                                );

};





#endif   // _ELISE_XML_GEN_MMBY_P_



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
