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



#include "StdAfx.h"






// En entree c'est le clip et la taille, d'entree
// en sortie c'est la translation (diff si le scale<0) et la taille
// de sortie

cXML_ParamNuage3DMaille CropAndSousEch
                        (
                             const cXML_ParamNuage3DMaille & anInit,
                             Pt2dr & aP0,
                             double aSc,
                             Pt2dr & aSz
                        )
{
    cXML_ParamNuage3DMaille aRes = anInit;
    aRes.AttributsNuage3D().clear();

    aRes.SsResolRef().SetVal(aRes.SsResolRef().Val() *aSc);

    ElAffin2D anAff = ElAffin2D::TransfoImCropAndSousEch(aP0,aSc,&aSz);
    AddAffinite(aRes.Orientation(),anAff); // RPCNuage
    aRes.NbPixel() = round_ni(aSz);

    aSz =  Pt2dr(aRes.NbPixel());
    aP0 = anAff.inv()(Pt2dr(0,0));




    for
    (
          std::list<cVerifNuage>::iterator itVN=aRes.VerifNuage().begin();
          itVN != aRes.VerifNuage().end();
          itVN++
    )
    {
       itVN->IndIm() = anAff( itVN->IndIm() );
    }

    return aRes;
}

cXML_ParamNuage3DMaille CropAndSousEch
                        (
                             const cXML_ParamNuage3DMaille & anInit,
                             Pt2dr & aP0,
                             double aSc
                        )
{
   Pt2dr aSz = Pt2dr(anInit.NbPixel());
   return CropAndSousEch(anInit,aP0,aSc, aSz);
}

/***********************************************/
/*                                             */
/*              cArgAuxBasc                    */
/*                                             */
/***********************************************/

cArgAuxBasc::cArgAuxBasc(Pt2di aSz) :
   mImInd     (aSz.x,aSz.y),
   mImTriInv  (aSz.x,aSz.y)
{
}

cArgAuxBasc_Sec::cArgAuxBasc_Sec(Pt2di aSz) :
   cArgAuxBasc (aSz),
   mImZ        (aSz.x,aSz.y,-1e10)
{
}


/***********************************************/
/*                                             */
/*              cGrpeLayerN3D                  */
/*                                             */
/***********************************************/

cGrpeLayerN3D::cGrpeLayerN3D(int aK1,int aK2,const std::string & aName) :
   mName (aName),
   mK1   (aK1),
   mK2   (aK2)
{
}

/***********************************************/
/*                                             */
/*             cLayerNuage3DM                  */
/*                                             */
/***********************************************/


std::string NamePlyOfType(GenIm::type_el aType)
{
   if (aType==GenIm::u_int1) return "uchar";
   if (aType==GenIm::real4) return "float";
   if (aType==GenIm::real8) return "float64";

   ELISE_ASSERT(false,"NamePlyOfType");
    return "";
}

cLayerNuage3DM::cLayerNuage3DM(Im2DGen * anIm, const std::string & aName) :
 mIm   (anIm),
 mName (aName)
{
}

void cLayerNuage3DM::PlyPutHeader(FILE * aFp) const
{
    std::string aType= NamePlyOfType(mIm->TypeEl());
    fprintf(aFp,"property %s %s\n",aType.c_str(),mName.c_str());
}


void cLayerNuage3DM::PlyPutData(FILE * aFP,const Pt2di & anI,bool aModeBin) const
{
  if (!aModeBin)
     fprintf(aFP," ");
   mIm->PutData(aFP,anI,aModeBin);
}


Im2DGen * cLayerNuage3DM::Im() const
{
   return mIm;
}

const std::string & cLayerNuage3DM::Name() const
{
   return mName;
}


bool  cLayerNuage3DM::Compatible(const cLayerNuage3DM & anA2) const
{
   return    (mName==anA2.mName)
          && (mIm->TypeEl()==anA2.mIm->TypeEl());
}


cLayerNuage3DM::~cLayerNuage3DM()
{
   delete mIm;
}

/***********************************************/
/*                                             */
/*             cElNuage3DMaille                */
/*                                             */
/***********************************************/

cBasicGeomCap3D * DeformCameraAffine
                  (
                        const cAffinitePlane & aXmlApInit2Cur,
                        cBasicGeomCap3D * aCam0,
                        const std::string & aName,
                        const std::string &aNameIma
                   );


Pt2di TTT(const Pt2di & aP)
{
   std::cout << "TTTTTTTTTTtt " << aP << "\n";
   return aP;
}
const std::string&  cElNuage3DMaille::NameFile() const {return mNameFile;}

// static int TheTypeUnknown eTIGB_Unknown;

cElNuage3DMaille::cElNuage3DMaille
(
    const std::string &             aDir,
    const cXML_ParamNuage3DMaille & aParam,
    Fonc_Num aFDef,
    const std::string & aNameFile,
    bool     WithEmpyData
) :
   mITypeCam      (eTIGB_Unknown),
   mEmptyData     (WithEmpyData),
   mDir           (aDir),
   mICNM          (cInterfChantierNameManipulateur::StdAlloc(0,0,mDir,cTplValGesInit<std::string>(),0)),
   mParams        (*(new cXML_ParamNuage3DMaille(aParam))),
   mSzGeom        (mParams.NbPixel()),
   mSzData        (WithEmpyData ? Pt2di(1,1) : mSzGeom),
   mImDef         (mSzData.x,mSzData.y,0),
   mNbPts         (0),
   mTImDef        (mImDef),
   mImDefInterp   (mSzData.x,mSzData.y,0),
   mTImDefInterp  (mImDefInterp),
   mCam           (   // RPCNuage
                       mParams.NameOri().IsInit()                                    ?
                       DeformCameraAffine
                       (
                           mParams.Orientation().OrIntImaM2C().Val(),
                           cBasicGeomCap3D::StdGetFromFile(mParams.NameOri().Val(),mITypeCam),
                           aNameFile,
                           aDir+mParams.PN3M_Nuage().Image_Profondeur().Val().Image() // Ne Sert que pour la taille ?
                       )  :  
                       Cam_Gen_From_XML(mParams.Orientation(),mICNM,aNameFile)     
                  ),
   mImEtire       (1,1),
   mVoisImDef     (mImDef),
   mTVoisImDef    (mVoisImDef),
   mNormByCenter  (0),
   mDistCenter    (0),
   m2RepGlob      (0),
   m2RepLoc       (0),
   mAnam          (0),
   mGenerMesh     (false),
   mNumPts        (1,1),
   mTNumP         (mNumPts),
   mNbTri         (0),
   mResolGlobCalc (false),
   mResolGlob     (0),
   mNameFile      (aNameFile)

{



    if (aParam.RepereGlob().IsInit())
    {
         m2RepGlob = new cChCoCart(cChCoCart::Xml2El(aParam.RepereGlob().Val()));
         m2RepLoc  = new cChCoCart (m2RepGlob->Inv());
    }
    if (aParam.Anam().IsInit())
    {
         mAnam = cInterfSurfaceAnalytique::FromXml(aParam.Anam().Val());
    }
    ELISE_ASSERT
    (
         !((m2RepGlob)&&(mAnam)),
         "Both RepereGlob and Anam in  cXML_ParamNuage3DMaille"
    );
/*
*/


    ELISE_COPY
    (
         mImDef.all_pts(),
         aFDef>0,
         mImDef.out() | sigma(mNbPts)
    );
    Pt2di aP;
    for (aP.x = 0 ; aP.x<mSzData.x-1 ; aP.x++)
    {
        for (aP.y = 0 ; aP.y<mSzData.y-1 ; aP.y++)
        {
              UpdateDefInterp(aP);
        }
    }
    for
    (
       std::list<cAttributsNuage3D>::const_iterator itA=aParam.AttributsNuage3D().begin();
       itA!=aParam.AttributsNuage3D().end();
       itA++
    )
    {
        Std_AddAttrFromFile
        (
           (itA->AddDir2Name().Val() ? aDir : "") + itA->NameFileImage(),
           itA->Dyn().Val(),
           itA->Scale().Val()
        );
    }
}

void cElNuage3DMaille::SetNbPts(int val)
{
    mNbPts = val;
}
int cElNuage3DMaille::GetNbPts()
{
    return mNbPts;
}
void cElNuage3DMaille::SetVoisImDef(Im2D_Bits<1> anIm)
{
    mVoisImDef = anIm;
    mTVoisImDef = TIm2DBits<1> (mVoisImDef);
}

void cElNuage3DMaille::SetNormByCenter(int val)
{
    mNormByCenter = val;
}

void cElNuage3DMaille::SetDistCenter(double val)
{
    mDistCenter = val;
}

bool cElNuage3DMaille::IsEmpty()
{
    Pt2di aP;
    for (aP.x=0 ; aP.x < mSzData.x ; aP.x++)
        for ( aP.y=0 ; aP.y < mSzData.y; aP.y++)
            if (mTImDef.get(aP))
               return false;

    return true;
}

Pt2di    cElNuage3DMaille::SzBasicCapt3D() const
{
   return SzGeom();
}


bool  cElNuage3DMaille::CaptHasData(const Pt2dr & aP) const
{
    return IndexHasContenuForInterpol(aP);
}

void cElNuage3DMaille::AssertNoEmptyData() const
{
    ELISE_ASSERT(!mEmptyData,"cElNuage3DMaille::AssertNoEmptyData");
}

const Pt2di &  cElNuage3DMaille::SzUnique() const
{
    AssertNoEmptyData();
    return mSzGeom;
}

ElSeg3D  cElNuage3DMaille::Capteur2RayTer(const Pt2dr & aP) const
{
    AssertCamInit();
    return FaisceauFromIndex(aP);
}

// En fait peut avoir valeur par defaut

double cElNuage3DMaille::ResolSolOfPt(const Pt3dr & aP) const
{
    Pt2dr aPIm = Ter2Capteur(aP);
    ElSeg3D aSeg = Capteur2RayTer(aPIm+Pt2dr(1,0));
    return aSeg.DistDoite(aP);
}

double cElNuage3DMaille::ResolSolGlob() const
{
   AssertNoEmptyData();
   if (! mResolGlobCalc)
   {
      double aSomRes = 0.0;
      double aSomPds = 0.0;

      for (int aNb=50 ; (aNb>0) &&  (aSomPds<500) ; aNb = ElMin(aNb-1,round_ni(aNb/1.2)))
      {
          Pt2di aDec (aNb,aNb);
          Pt2di aP;
          for (aP.x =0 ; aP.x<mSzData.x ; aP.x+=aDec.x)
          {
              for (aP.y =0 ; aP.y<mSzData.y ; aP.y+=aDec.y)
              {
                   if (IndexHasContenu(aP))
                   {
                        aSomPds++;
                        aSomRes += ResolSolOfPt(PtOfIndex(aP));
                   }
              }
          }
      }

      mResolGlobCalc = true;
      if (aSomPds==0)
      {
          std::cout << "Masq=" << mParams.PN3M_Nuage().Image_Profondeur().Val().Masq() << "\n";
          Tiff_Im::Create8BFromFonc("PbMasq.tif",mSzData,ImMask().in());
          ELISE_ASSERT(aSomPds!=0.0,"cElNuage3DMaille::ResolSolGlob");
      }
      mResolGlob = aSomRes / aSomPds;
   }

   return mResolGlob;
}

double  cElNuage3DMaille::ResolImRefFromCapteur() const
{
    return  mParams.SsResolRef().Val();
}

Pt2dr cElNuage3DMaille::ImRef2Capteur   (const Pt2dr & aP) const
{
    return aP / mParams.SsResolRef().Val();
}


bool   cElNuage3DMaille::HasRoughCapteur2Terrain() const
{
    return true;
}


bool  cElNuage3DMaille::HasPreciseCapteur2Terrain() const
{
    return true;
}

Pt3dr cElNuage3DMaille::RoughCapteur2Terrain   (const Pt2dr & aP) const
{
    return PtOfIndexInterpol(aP);
}


Pt3dr cElNuage3DMaille::PreciseCapteur2Terrain   (const Pt2dr & aP) const
{
    return PtOfIndexInterpol(aP);
}

Pt2dr cElNuage3DMaille::Ter2Capteur(const Pt3dr & aP) const
{
    return Terrain2Index(aP);
}


bool cElNuage3DMaille::PIsVisibleInImage   (const Pt3dr & aP,cArgOptionalPIsVisibleInImage  *anArg) const
{
    return   mCam->PIsVisibleInImage (aP,anArg);
}


void cElNuage3DMaille::AddTri(std::vector<tTri> & aMesh,const tIndex2D & aP,int *K123,int anOffset) const
{
    for (int aK=0 ; aK<3 ; aK++)
       if (!mTImDef.get(aP+VOIS_9[K123[aK]],0))
          return;

     aMesh.push_back
       (
            Pt3di
            (
                  anOffset+ mTNumP.get(aP+VOIS_9[K123[0]]),
                  anOffset+ mTNumP.get(aP+VOIS_9[K123[1]]),
                  anOffset+ mTNumP.get(aP+VOIS_9[K123[2]])
            )
       );
}

double cElNuage3DMaille::TriArea(const Pt3dr &aP1,const Pt3dr &aP2, const Pt3dr &aP3) const
{
  double d1 = aP1.x*(aP2.y-aP3.y)+aP2.x*(aP3.y-aP1.y)+aP3.x*(aP1.y-aP2.y);
  double d2 = aP1.y*(aP2.z-aP3.z)+aP2.y*(aP3.z-aP1.z)+aP3.y*(aP1.z-aP2.z);
  double d3 = aP1.z*(aP2.x-aP3.x)+aP2.z*(aP3.x-aP1.x)+aP3.z*(aP1.x-aP2.x);
  double a = sqrt(Square(d1)+Square(d2)+Square(d3))/2.;

  return a;
}

void cElNuage3DMaille::GenTri(std::vector<tTri> & aMesh,const tIndex2D &aP,int aOffset) const
{
    aMesh.clear();

    int aKT[4] = {4,5,7,8};
    std::vector<Pt3dr>  aPT;
    for (int aK=0 ; aK<4 ; aK++)
    {
      if (!mTImDef.get(aP+VOIS_9[aKT[aK]],0))
          return;
      aPT.push_back(PtOfIndex(aP+VOIS_9[aKT[aK]]));
    }

    double aT1 = TriArea(aPT[0],aPT[1],aPT[2]);
    double aT2 = TriArea(aPT[1],aPT[2],aPT[3]);
    double aT3 = TriArea(aPT[2],aPT[3],aPT[0]);
    double aT4 = TriArea(aPT[3],aPT[0],aPT[1]);

    int aKT1[3]={7,5,4};
    int aKT2[3]={5,7,8};
    if (aT1+aT2>aT3+aT4) //The best diagonal is the one that minimize the area of the 2 triangles
    {
      aKT1[1] = 8;
      aKT2[1] = 4;
    }

    AddTri(aMesh,aP,aKT1,aOffset);
    AddTri(aMesh,aP,aKT2,aOffset);
}

void cElNuage3DMaille::AddExportMesh()
{
   AssertNoEmptyData();
   mGenerMesh = true;
   mNbTri = 0;
   mNumPts.Resize(mSzData);
   ELISE_COPY(mNumPts.all_pts(),-1,mNumPts.out());
   mTNumP = TIm2D<INT4,INT>(mNumPts);
   int aNum = 0;
   for (tIndex2D anI=Begin(); anI!=End() ;IncrIndex(anI))
   {
        mTNumP.oset(anI,aNum++);
   }

   for (tIndex2D anI=Begin(); anI!=End() ;IncrIndex(anI))
   {
      std::vector<tTri>  aVT;
      GenTri(aVT,anI,0);
      mNbTri += (int)aVT.size();
   }

   // std::cout << "NB FACE " << mNbTri  << " NB SOM " << mNbPts << "\n";
}

void cElNuage3DMaille::PlyPutDataOneFace(FILE * aFP,const tTri& aTr, bool aModeBin) const
{
   if (aModeBin)
   {
       U_INT1 aNb3 = 3;
       WriteType(aFP,aNb3);
       WriteType(aFP,int(aTr.x));
       WriteType(aFP,int(aTr.y));
       WriteType(aFP,int(aTr.z));

   }
   else
   {
        fprintf(aFP,"3 %d %d %d\n",aTr.x,aTr.y,aTr.z);
   }
}

void cElNuage3DMaille::PlyPutDataFace(FILE * aFP,bool aModeBin,int & anOffset) const
{
   if (! mGenerMesh)
       return;
   for (tIndex2D anI=Begin(); anI!=End() ;IncrIndex(anI))
   {
      std::vector<tTri>  aVT;
      GenTri(aVT,anI,anOffset);

      for(int aKT=0 ; aKT<int(aVT.size()) ; aKT++)
      {
           PlyPutDataOneFace(aFP,aVT[aKT],aModeBin);
      }
   }
   anOffset += mNbTri;
}





Im2D_Bits<1>   cElNuage3DMaille::ImDef()
{
   return mImDef;
}


Pt3dr  cElNuage3DMaille::Loc2Glob(const Pt3dr & aP) const
{
    if (m2RepGlob)
       return m2RepGlob->FromLoc(aP);
    if (mAnam)
       return mAnam->UVL2E(aP);

    return aP;
}


Pt3dr  cElNuage3DMaille::Glob2Loc(const Pt3dr & aP) const
{
    if (m2RepLoc)
       return m2RepLoc->FromLoc(aP);
    if (mAnam)
       return mAnam->E2UVL(aP);

    return aP;
}

cBasicGeomCap3D *   cElNuage3DMaille::Cam() const
{
   return mCam;
}

cElNuage3DMaille::~cElNuage3DMaille()
{
   DeleteAndClear (mAttrs);
   delete mCam;
}


void cElNuage3DMaille::Save(const std::string & aName)
{
  mDir = DirOfFile(aName);

   std::string    aNameM= NameWithoutDir(aName + "_Masq.tif");
   std::string    aNameP= NameWithoutDir(aName + "_Prof.tif");
   if ( mParams.Image_Point3D().IsInit())
   {
       mParams.Image_Point3D().Val().Image() = aNameP;
       mParams.Image_Point3D().Val().Masq() = aNameM;
   }
   else if ( mParams.Image_Profondeur().IsInit())
   {
       mParams.Image_Profondeur().Val().Image() = aNameP;
       mParams.Image_Profondeur().Val().Masq() = aNameM;
   }
/*
   else if (mParams.Image_MNT().IsInit())
   {
       mParams.Image_MNT().Val().Image() = aNameP;
       mParams.Image_MNT().Val().Masq() = aNameM;
   }
*/
   else
   {
     ELISE_ASSERT(false,"cElNuage3DMaille::Save");
   }


   Tiff_Im::CreateFromIm(mImDef,mDir+aNameM);
   V_Save(aNameP);


   cXML_ParamNuage3DMaille aParam = mParams;
   for (int aKG=0 ; aKG<int(mGrpAttr.size()) ; aKG++)
   {
       const cGrpeLayerN3D & aGrp =  mGrpAttr[aKG];
       std::string aNameG = aName + "_" + StdPrefix(aGrp.mName) + std::string(".tif");
       cAttributsNuage3D anAttr;
       anAttr.NameFileImage() = aNameG;
       aParam.AttributsNuage3D().push_back(anAttr);

       std::vector<Im2DGen> aVIm;
       for (int aKIm=aGrp.mK1 ; aKIm <aGrp.mK2 ; aKIm++)
       {
           aVIm.push_back(*(mAttrs[aKIm]->Im()));
       }
       Tiff_Im::CreateFromIm(aVIm,mDir+aNameG);
   }


   MakeFileXML(aParam,aName+".xml");

}

bool  cElNuage3DMaille::Compatible(const cElNuage3DMaille & aN2) const
{
   if (mAttrs.size() != aN2.mAttrs.size())
      return false;
   for (int aKA=0 ; aKA<int(mAttrs.size()) ; aKA++)
       if (! mAttrs[aKA]->Compatible(*aN2.mAttrs[aKA]))
           return false;
   return true;
}

cElNuage3DMaille::tIndex2D  cElNuage3DMaille::Begin() const
{
     Pt2di aP(0,0);
     if (! IndexHasContenu(aP))
        IncrIndex(aP);
     return aP;
}

void cElNuage3DMaille::PlyPutFile
     (
           const std::string & aName,
           const std::list<std::string> & aComments,
           bool aModeBin,
           bool SavePtsCol,
           int aAddNormale,
           const std::list<std::string> & aNormName,
           bool DoublePrec,
           const Pt3dr& anOffset
     ) const
{
    std::vector<const cElNuage3DMaille *> aVN;
    aVN.push_back(this);
    PlyPutFile(aName,aComments,aVN,0,0,aModeBin, SavePtsCol, aAddNormale, aNormName, DoublePrec, anOffset);
}



void cElNuage3DMaille::PlyPutFile
     (
           const std::string & aName,
           const std::list<std::string> & aComments,
           const std::vector<const cElNuage3DMaille *> & aVN,
           const std::vector<Pt3dr> * mPts,
           const std::vector<Pt3di> * mCouls,
           bool aModeBin,
           bool SavePtsCol,
           int aAddNormale,
           const std::list<std::string> & aNormName,
           bool DoublePrec,
           const Pt3dr & anOffset
     )
{
   std::string aTypeXYZ = DoublePrec ? "float64" : "float";




   int aNbF = 0;
   int aNbS = 0;
   if (mPts)
   {
       aNbS = (int)mPts->size();
       ELISE_ASSERT(mCouls,"Pts sans coul dans cElNuage3DMaille::PlyPutFile");
       ELISE_ASSERT(aNbS==int(mPts->size()),"Pts and coul dif size in PlyPutFile");
   }
   const cElNuage3DMaille * aN0 = 0;
   int aNbAttr = 0;

   if (aVN.size()==0)
   {
      ELISE_ASSERT(mPts!=0,"No Nuage in cElNuage3DMaille::PlyPutFile");
      aNbAttr = 3;
   }
   else
   {
      aN0 = aVN[0];
      aNbAttr = (int)aN0->mAttrs.size();
   }
   for (int aK=0 ; aK<int(aVN.size()) ; aK++)
   {
       aNbS += aVN[aK]->mNbPts;
       aNbF += aVN[aK]->mNbTri;
       if (aK>=1)
       {
          ELISE_ASSERT
          (
              aN0->Compatible(*aVN[aK]),
              "Incompatible Nuage in cElNuage3DMaille::PlyPutFile"
          );
       }
   }

   //GERALD : Correction bug : ecriture binaire incorrect sous windows
   //Mode Ecriture : binaire ou non
   std::string mode = aModeBin ? "wb" : "w";
   FILE * aFP = FopenNN(aName,mode,"cElNuage3DMaille::PlyPutFile");

   //Header
   fprintf(aFP,"ply\n");
   std::string aBinSpec =       MSBF_PROCESSOR() ?
                          "binary_big_endian":
                          "binary_little_endian" ;

   fprintf(aFP,"format %s 1.0\n",aModeBin?aBinSpec.c_str():"ascii");

   for
   (
        std::list<std::string>::const_iterator itS=aComments.begin();
        itS!=aComments.end();
        itS++
   )
   {
      fprintf(aFP,"comment %s\n",itS->c_str());
   }
   fprintf(aFP,"element vertex %d\n",aNbS);
   fprintf(aFP,"property %s x\n",aTypeXYZ.c_str());
   fprintf(aFP,"property %s y\n",aTypeXYZ.c_str());
   fprintf(aFP,"property %s z\n",aTypeXYZ.c_str());

   if (aAddNormale)
   {
       if (aNormName.size()==3)
       {
           for (std::list<std::string>::const_iterator itS=aNormName.begin(); itS!=aNormName.end(); itS++)
           {
               fprintf(aFP,"property %s %s\n",aTypeXYZ.c_str(),itS->c_str());
           }
       }
       else{
           if (aNormName.size()!=0)
           {
               std::cout << "Warning NormName not used, should be 3 strings"<<std::endl;
           }
           fprintf(aFP,"property %s nx\n",aTypeXYZ.c_str());
           fprintf(aFP,"property %s ny\n",aTypeXYZ.c_str());
           fprintf(aFP,"property %s nz\n",aTypeXYZ.c_str());
       }
   }

   const char * aVCoul[3]={"red","green","blue"};
   for (int aK=0 ; aK<aNbAttr ; aK++)
   {
       if (aN0)
       {
           aN0->mAttrs[aK]->PlyPutHeader(aFP);
       }
       else
       {
           if (SavePtsCol)
             {fprintf(aFP,"property uchar %s\n",aVCoul[aK]);}
       }
   }

   fprintf(aFP,"element face %d\n",aNbF);
   fprintf(aFP,"property list uchar int vertex_indices\n");
   fprintf(aFP,"end_header\n");

   //Data
   for (int aK=0 ; aK<int(aVN.size()) ; aK++)
      aVN[aK]-> PlyPutDataVertex(aFP,aModeBin, aAddNormale,DoublePrec,anOffset);

   if (mPts)
   {
      for (int aKV=0 ; aKV<int(mPts->size()) ; aKV++)
      {
          Pt3dr aP = (*mPts)[aKV];
          Pt3di aC = (*mCouls)[aKV];
          if (aModeBin)
          {
             WriteType(aFP,float(aP.x));
             WriteType(aFP,float(aP.y));
             WriteType(aFP,float(aP.z));
             if (SavePtsCol)
             {
                WriteType(aFP,(U_INT1)(aC.x));
                WriteType(aFP,(U_INT1)(aC.y));
                WriteType(aFP,(U_INT1)(aC.z));
             }
          }
          else
          {
              if (SavePtsCol)
                {
                  fprintf(aFP,"%.3f %.3f %.3f %d %d %d\n",aP.x,aP.y,aP.z,aC.x,aC.y,aC.z);
                }
              else
                { fprintf(aFP,"%.3f %.3f %.3f\n",aP.x,aP.y,aP.z); }
          }
      }
   }

   int anOffsetNbTri = 0;
   for (int aK=0 ; aK<int(aVN.size()) ; aK++)
   {
      aVN[aK]-> PlyPutDataFace(aFP,aModeBin,anOffsetNbTri);
   }

   ElFclose(aFP);
}


void cElNuage3DMaille::NuageXZGCOL(const std::string & aName,bool B64)
{
   AssertNoEmptyData();
   std::string aNameXYZ = aName+"_XYZ.tif";
   L_Arg_Opt_Tiff aL;
   aL = aL+Arg_Tiff(Tiff_Im::ANoStrip());

   Tiff_Im aXYZ(aNameXYZ.c_str(),mSzData, B64 ? GenIm::real8 : GenIm::real4,Tiff_Im::No_Compr,Tiff_Im::RGB,aL);

   Im2D_REAL8 aImX(mSzData.x,mSzData.y,0.0);
   Im2D_REAL8 aImY(mSzData.x,mSzData.y,0.0);
   Im2D_REAL8 aImZ(mSzData.x,mSzData.y,0.0);

   Pt2di anI;
   for (anI.x=0 ; anI.x<mSzData.x ; anI.x++)
   {
      for (anI.y=0 ; anI.y<mSzData.y ; anI.y++)
      {
           if (IndexHasContenu(anI))
            {
                Pt3dr aP3 = PtOfIndex(anI);
                aImX.SetR(anI,aP3.x);
                aImY.SetR(anI,aP3.y);
                aImZ.SetR(anI,aP3.z);
            }
      }
  }
  ELISE_COPY
  (
      aXYZ.all_pts(),
      Virgule(aImX.in(),aImY.in(),aImZ.in()),
      aXYZ.out()
  );
}



void cElNuage3DMaille::PlyPutDataVertex(FILE * aFP, bool aModeBin, int aAddNormale,bool DoublePrec,const Pt3dr & anOffset) const
{
    if (aAddNormale)
    {
    ELISE_ASSERT((aAddNormale%2) && (aAddNormale>2),"cElNuage3DMaille::NormaleOfIndex: wSize should be an odd > 1 (3, 5, 7...)");
    }

    for (tIndex2D anI=Begin(); anI!=End() ;IncrIndex(anI))
    {
           Pt3dr aP = PtOfIndex(anI) - anOffset;
           // std::cout << "PlyPutData:::: " << aP << "\n"; getchar();

           if (DoublePrec)
           {
                double xyz[3];
                xyz[0] = aP.x;
                xyz[1] = aP.y;
                xyz[2] = aP.z;
                if (aModeBin)
                {
                    int aNb= (int)fwrite(xyz,sizeof(double),3,aFP);
                    ELISE_ASSERT(aNb==3,"cElNuage3DMaille::PlyPutDataVertex");
                }
                else
                {
                   fprintf(aFP,"%.6f %.6f %.6f", xyz[0], xyz[1], xyz[2]);
                }
           }
           else
           {
                float xyz[3];
                xyz[0] = (float)aP.x;
                xyz[1] = (float)aP.y;
                xyz[2] = (float)aP.z;
                if (aModeBin)
                {
                    int aNb = (int)fwrite(xyz, sizeof(float), 3, aFP);
                    ELISE_ASSERT(aNb==3,"cElNuage3DMaille::PlyPutDataVertex");
                }
                else
                {
                   fprintf(aFP,"%.6f %.6f %.6f", xyz[0], xyz[1], xyz[2]);
                }
           }

       if (aAddNormale)
       {
           Pt3dr aN = NormaleOfIndex(anI, aAddNormale, anOffset);

           if (DoublePrec)
           {
               double Nxyz[3];
               Nxyz[0] = (double)aN.x;
               Nxyz[1] = (double)aN.y;
               Nxyz[2] = (double)aN.z;

               if (aModeBin)
               {
                   int aNb= (int)fwrite(Nxyz, sizeof(double), 3, aFP);
                   ELISE_ASSERT(aNb==3,"cElNuage3DMaille::PlyPutDataVertex-Normale");
               }
               else
               {
                   fprintf(aFP," %.6f %.6f %.6f", Nxyz[0], Nxyz[1], Nxyz[2]);
               }
           }
           else
           {
               float Nxyz[3];
               Nxyz[0] = (float)aN.x;
               Nxyz[1] = (float)aN.y;
               Nxyz[2] = (float)aN.z;

               if (aModeBin)
               {
                   int aNb= (int)fwrite(Nxyz, sizeof(float), 3, aFP);
                   ELISE_ASSERT(aNb==3,"cElNuage3DMaille::PlyPutDataVertex-Normale");
               }
               else
               {
                   fprintf(aFP," %.6f %.6f %.6f", Nxyz[0], Nxyz[1], Nxyz[2]);
               }
           }
       }

           for (int aK=0 ; aK<int(mAttrs.size()) ; aK++)
           {
              mAttrs[aK]->PlyPutData(aFP,anI,aModeBin);
           }
           if (!aModeBin)
              fprintf(aFP,"\n");
    }
}

       // -----------   ITERATIONS ---------------

cElNuage3DMaille::tIndex2D  cElNuage3DMaille::End() const
{
     return Pt2di(0,mSzData.y);
}

void  cElNuage3DMaille::IncrIndSsFiltre(tIndex2D & aP) const
{
    aP.x++;
    if (aP.x==mSzData.x)
    {
       aP.y++;
       aP.x=0;
    }
}

void  cElNuage3DMaille::IncrIndex(tIndex2D & aP) const
{
    IncrIndSsFiltre(aP);
    while (aP.y<mSzData.y)
    {
        if (IndexHasContenu(aP)) return;
        IncrIndSsFiltre(aP);
    }
}


       // -----------   MODIFICATIONS  ---------------

void  cElNuage3DMaille::SetPtOfIndex(const tIndex2D & anIndex,const Pt3dr & aP3)
{
    AssertInsideData(anIndex);
    mTImDef.oset(anIndex,1);
    V_SetPtOfIndex(anIndex,Glob2Loc(aP3));
    UpdateVoisAfterModif(anIndex);
}
void  cElNuage3DMaille::SetNoValue(const tIndex2D & anIndex)
{
    AssertInsideData(anIndex);
    mTImDef.oset(anIndex,0);
    UpdateVoisAfterModif(anIndex);
}


void cElNuage3DMaille::UpdateDefInterp(const Pt2di & aP)
{
     mTImDefInterp.oset
     (
        aP,
            mTImDef.get(aP+Pt2di(0,0),0)
         && mTImDef.get(aP+Pt2di(0,1),0)
         && mTImDef.get(aP+Pt2di(1,0),0)
         && mTImDef.get(aP+Pt2di(1,1),0)
     );
}
void cElNuage3DMaille::UpdateVoisAfterModif(const Pt2di & aP)
{
   Pt2di aQ;

   for (aQ.x=aP.x-1 ; aQ.x<=aP.x; aQ.x++)
   {
       for (aQ.y=aP.y-1 ; aQ.y<=aP.y; aQ.y++)
       {
           if (IndexInsideData(aQ))
           {
               UpdateDefInterp(aQ);
           }
       }
   }
}

       // -----------   ASSERTION   ---------------

void   cElNuage3DMaille::AssertInsideData(const tIndex2D & anI) const
{
   ELISE_ASSERT(IndexInsideData(anI),"cElNuage3DMaille::AssertInside");
}
void   cElNuage3DMaille::AssertInsideGeom(const tIndex2D & anI) const
{
   ELISE_ASSERT(IndexInsideGeom(anI),"cElNuage3DMaille::AssertInside");
}

void   cElNuage3DMaille::AssertCamInit() const
{
   ELISE_ASSERT(mCam!=0,"cElNuage3DMaille::AssertCamInit");
}


/*
*/
       // -----------   AUTRES   ---------------


Pt3dr cElNuage3DMaille::Loc_PtOfIndexInterpol(const Pt2dr & aPR) const
{
   Pt2di aPI = round_down(aPR);
   double aP1X = aPR.x - aPI.x;
   double aP1Y = aPR.y - aPI.y;
   double aP0X = 1-aP1X;
   double aP0Y = 1-aP1Y;

   return   Loc_PtOfIndex(aPI+Pt2di(0,0))*aP0X*aP0Y
          + Loc_PtOfIndex(aPI+Pt2di(1,0))*aP1X*aP0Y
          + Loc_PtOfIndex(aPI+Pt2di(0,1))*aP0X*aP1Y
          + Loc_PtOfIndex(aPI+Pt2di(1,1))*aP1X*aP1Y ;
}

Pt3dr cElNuage3DMaille::PtOfIndex(const tIndex2D & aPR) const
{
    return Loc2Glob(Loc_PtOfIndex(aPR));
}
Pt3dr cElNuage3DMaille::PtOfIndexInterpol(const Pt2dr & aPR) const
{
    return Loc2Glob(Loc_PtOfIndexInterpol(aPR));
}
Pt3dr cElNuage3DMaille::IndexAndProf2Euclid(const   Pt2dr & aP,const double & aProf) const
{
   return Loc2Glob(Loc_IndexAndProf2Euclid(aP,aProf));
}
Pt3dr cElNuage3DMaille::Euclid2ProfAndIndex(const   Pt3dr & aP) const
{
   return Loc_Euclid2ProfAndIndex(Glob2Loc(aP));
}
Pt3dr cElNuage3DMaille::IndexAndProfPixel2Euclid(const   Pt2dr & aP,const double & aProf) const
{
   return Loc2Glob(Loc_IndexAndProfPixel2Euclid(aP,aProf));
}

Pt3dr cElNuage3DMaille::Euclid2ProfPixelAndIndex(const   Pt3dr & aP) const
{
    return Loc_Euclid2ProfPixelAndIndex(Glob2Loc(aP));
}





double cElNuage3DMaille::DiffDeSurface
       (
           bool &                    isOk,
           const tIndex2D&           anI1,
           const cElNuage3DMaille &  aN2
       ) const
{
    aN2.AssertCamInit();

    isOk = false;
    if (IndexIsOK(anI1))
    {
        Pt3dr  aPT1 = PtOfIndex(anI1);
        Pt2dr  anI2  = aN2.mCam->Ter2Capteur(aPT1);
        if (aN2.IndexIsOKForInterpol(anI2))
        {
            isOk = true;
// std::cout << anI1 << anI2 << "\n";
            Pt3dr  aPT2 = aN2.PtOfIndexInterpol(anI2);
            ElSeg3D aSeg = aN2.mCam->Capteur2RayTer(anI2);
            return scal(aSeg.TgNormee(),aPT1-aPT2);
        }
    }

    return -1e9;
}

//Compute local normal on 3D points in a window (wSize x wSize)
Pt3dr cElNuage3DMaille::NormaleOfIndex(const tIndex2D& anI1, int wSize, const Pt3dr & anOffset) const
{
    if (IndexHasContenu(anI1))
    {
                double aFact = 0.1;
        Pt2dr anI1r(anI1.x, anI1.y);
        ElSeg3D aV = Capteur2RayTer(anI1r);
        Pt3dr aTgt = aV.TgNormee();
                if (mNormByCenter==1)
                   return aTgt * (-aFact);
                else if (mNormByCenter==2)
                {
                    Pt3dr aCentreOptique = mCam->OrigineProf() - anOffset;

                    if(aCentreOptique.z==0) //On est probablement en mode grille, donc il faut calculer un pseudo centre...
                    {
                        std::cout<<"Be carefull, GRID orientation -> it's not the real center. Camera distance set to "<<mDistCenter<<"m (DistC"<<std::endl;
                        aCentreOptique = aV.P0()+aTgt*mDistCenter - anOffset;
                    }
                    return aCentreOptique;
                }

        std::vector<Pt3dr> aVP;
        std::vector<double> aVPds;

        int halfSize  = wSize/2;
        tIndex2D anI2 = anI1 - Pt2di(halfSize, halfSize); //top-left corner of window
        tIndex2D anI3;

        //recherche des voisins et stockage des points
        for(int aK = 0; aK<wSize ; ++aK)
        {
            for(int bK = 0; bK<wSize ; ++bK)
            {
                anI3 = anI2 + Pt2di(aK, bK);

                if (IndexHasContenuAsNeighboor(anI3))
                {
                    aVP.push_back(PtOfIndex(anI3));
                    aVPds.push_back(1.f);
                }
            }
        }
                if (aVP.size() <= 4) return Pt3dr(0,0,0);

        //estimation du plan aux moindres carrés
        cElPlan3D aPlan(aVP, &aVPds);

        //retourne la normale en fonction de l'angle avec le segment PdV-Pt
        Pt3dr aN = aPlan.Norm()* aFact;
        // if (aN.x*aTgt.x + aN.y*aTgt.y + aN.z*aTgt.z < 0.f)
        if (scal(aN,aTgt) < 0.f)
        {
            return aN;
        }
        else
        {
            // return Pt3dr(-aN.x,-aN.y,-aN.z);
            return -aN;
        }
    }

    return Pt3dr(0.f,0.f,0.f);
}

cXML_ParamNuage3DMaille&  cElNuage3DMaille::Params()
{
   return mParams;
}

const cXML_ParamNuage3DMaille&  cElNuage3DMaille::Params()  const
{
   return mParams;
}


/*
CamStenope * cElNuage3DMaille::CamS()
{
   ELISE_ASSERT(mParams.Orientation().TypeProj().Val()==eProjStenope,"cElNuage3DMaille::CamStenope");
   return static_cast<CamStenope *> (mCam);
}
*/

Fonc_Num   cElNuage3DMaille::ReScaleAndClip(Fonc_Num aFonc,const Pt2dr & aP0,double aScale)
{
   return StdFoncChScale
          (
              aFonc,
              aP0,
              Pt2dr(aScale,aScale)
          );
}



cElNuage3DMaille * cElNuage3DMaille::ReScaleAndClip(double aScale)
{
   return ReScaleAndClip
          (
             Box2dr(Pt2dr(0,0),Pt2dr(mParams.NbPixel())),
             aScale
          );
}


cElNuage3DMaille * cElNuage3DMaille::ReScaleAndClip(Box2dr aBox,double aScale)
{
    Pt2dr aTr = aBox._p0;
    Pt2dr aSz = aBox.sz();
    Pt2di aSzI = round_up(aSz);

// std::cout << "ReScaleAndClippppp " << aScale << aTr << aSz <<"\n";
    cXML_ParamNuage3DMaille aNewParam = CropAndSousEch(mParams,aTr,aScale,aSz);

    Im2D_REAL4  aImPds(aSzI.x,aSzI.y);

    std::vector<Im2DGen*> aVNewAttr;
    std::vector<Im2DGen*> aVOldAttr;
    for (int aKA=0 ; aKA<int(mAttrs.size()) ; aKA++)
    {
        Im2DGen * anOld = mAttrs[aKA]->Im();
        aVOldAttr.push_back(anOld);
        aVNewAttr.push_back(anOld->ImOfSameType(aSzI));
    }

    cElNuage3DMaille * aRes = V_ReScale(aBox,aScale,aNewParam,aImPds,aVNewAttr,aVOldAttr);

    for (int aKA=0 ; aKA<int(mAttrs.size()) ; aKA++)
    {
        aRes->mAttrs.push_back(new cLayerNuage3DM(aVNewAttr[aKA],mAttrs[aKA]->Name()));
    }
    aRes->mGrpAttr = mGrpAttr;

    aRes->VerifParams();

/*
    for (int aKA=0 ; aKA<int(mAttrs.size()) ; aKA++)
    {
        Im2DGen * anOld = mAttrs[aKA]->Im();
        Im2DGen * aNew = anOld->ImOfSameType(Pt2di(aSz));

        Fonc_Num aF = ReScaleAndClip(anOld->in(0)*mImDef.in(0),aTr,aScale)
                      / Max(1e-5,aImPds.in(0));

        ELISE_COPY(aNew->all_pts(),aF,aNew->out());
        aRes->mAttrs.push_back(new cLayerNuage3DM(aNew,mAttrs[aKA]->Name()));
    }
    aRes->mGrpAttr = mGrpAttr;
*/



    if (aRes->mParams.Image_Point3D().IsInit())
    {
        aRes->mParams.Image_Point3D().Val().Image() = "";
        aRes->mParams.Image_Point3D().Val().Masq() = "";
    }
    else if (aRes->mParams.Image_Profondeur().IsInit())
    {
        aRes->mParams.Image_Profondeur().Val().Image() = "";
        aRes->mParams.Image_Profondeur().Val().Masq() = "";
    }
/*
    else if (aRes->mParams.Image_MNT().IsInit())
    {
        aRes->mParams.Image_MNT().Val().Image() = "";
        aRes->mParams.Image_MNT().Val().Masq() = "";
    }
*/
    else
    {
       ELISE_ASSERT(false,"cElNuage3DMaille::ReScaleAndClip");
    }

    // for (int aK=0 ; aK<
    return aRes;
}

void  cElNuage3DMaille:: VerifParams() const
{
}


void cElNuage3DMaille::AddGrpeLyaer(int aNb,const std::string & aName)
{
    mGrpAttr.push_back(cGrpeLayerN3D((int)mAttrs.size(), int(mAttrs.size()) + aNb, aName));
}

void cElNuage3DMaille::Std_AddAttrFromFile
     (
           const std::string &            aName,
           double aDyn,
           double aScale,
           bool   ForceRGB
     )
{
    std::vector<std::string> aVS;
    Tiff_Im aTF = Tiff_Im::UnivConvStd(aName);

    if ((aTF.phot_interp()==Tiff_Im::RGB) || ForceRGB)
    {
       AddGrpeLyaer(3,NameWithoutDir(aName));
       aVS.push_back("red");
       aVS.push_back("green");
       aVS.push_back("blue");
    }
    else if (aTF.phot_interp()==Tiff_Im::BlackIsZero)
    {
       AddGrpeLyaer(1,NameWithoutDir(aName));
       aVS.push_back("gray");
    }
    AddAttrFromFile(aName,0xFFFF,aVS,aDyn,aScale,ForceRGB);
}

void cElNuage3DMaille::AddAttrFromFile
     (
           const std::string &              aName,
           int                              aFlagChannel,
           const std::vector<std::string> & aNameProps,
           double aDyn,
           double aScale,
           bool ForceRGB
     )
{
    AssertNoEmptyData();
    Tiff_Im aTF = Tiff_Im::UnivConvStd(aName);
    GenIm::type_el aTEl = aTF.type_el();
    int aNbCOri= aTF.nb_chan();
    int aNbC = ForceRGB ? 3 : aNbCOri ;

    Output anOutGlog = Output::onul(1); // Initialisation par ce qu'il faut
    int aNbAdded = 0;
    int aNbProp = (int)aNameProps.size();
    for (int aK=0 ;aK<aNbC ; aK++)
    {
       Output anAdd =  Output::onul(1);
       if (aFlagChannel & (1<<aK))
       {
            Im2DGen * anI = Ptr_D2alloc_im2d(aTEl,mSzData.x,mSzData.y);
            anAdd  = anI->out();
            std::string aProp = "Unknown";
            if (aNbProp)
               aProp = aNameProps[ElMin(aNbAdded,aNbProp-1)];
            mAttrs.push_back(new cLayerNuage3DM(anI,aProp));

            aNbAdded++;
       }
       anOutGlog = (aK==0) ? anAdd : Virgule(anOutGlog,anAdd);
    }

    Fonc_Num aF = aTF.in_proj();


    if (aScale !=1)
    {
        aF = StdFoncChScale
             (
                 aF,
                 Pt2dr(0,0),
                 Pt2dr(aScale,aScale),
                 Pt2dr(1,1)
             );

    }
    if (aDyn !=1)
    {
        aF = aF * aDyn;
    }
    aF = Tronque(aTEl,aF);


    if (ForceRGB && (aNbCOri==1))
    {
        aF = Virgule(aF,aF,aF);
    }
    // ELISE_COPY(aTF.all_pts(),aF,anOutGlog);
    ELISE_COPY(rectangle(Pt2di(0,0),mSzData),aF,anOutGlog);
}

const std::vector<cLayerNuage3DM *> &  cElNuage3DMaille::Attrs() const
{
   return mAttrs;
}


bool cElNuage3DMaille::IndexIsPlani() const
{
    return true;
}

Pt2dr  cElNuage3DMaille::Index2Plani(const Pt2dr & aP) const
{
   return mCam->OrGlbImaC2M(aP);
}
Pt2dr  cElNuage3DMaille::Plani2Index(const Pt2dr & aP) const
{
   return mCam->OrGlbImaM2C(aP);
}


ElSeg3D cElNuage3DMaille::FaisceauFromIndex(const Pt2dr & aP) const
{
   ElSeg3D aRes = mCam->Capteur2RayTer(aP);
   return ElSeg3D(Loc2Glob(aRes.P0()),Loc2Glob(aRes.P1()));
}


Pt2dr   cElNuage3DMaille::Terrain2Index(const Pt3dr & aPt) const
{
    return mCam->Ter2Capteur(Glob2Loc(aPt));
}


double   cElNuage3DMaille::ProfEuclidOfIndex(const tIndex2D & anI) const
{
    return ProfOfIndex(anI);
}

void     cElNuage3DMaille::SetProfEuclidOfIndex(const tIndex2D & anI,double aProf)
{
   SetProfOfIndex(anI,aProf);
}


void   cElNuage3DMaille::SetProfOfIndex(const tIndex2D & aP,double)
{
   ELISE_ASSERT(false,"cElNuage3DMaille::SetProfOfIndex");
}

bool   cElNuage3DMaille::SetProfOfIndexIfSup(const tIndex2D & aP,double)
{
   ELISE_ASSERT(false,"cElNuage3DMaille::SetProfOfIndexIfSup");
   return false;
}



double  cElNuage3DMaille::ProfOfIndex(const tIndex2D & aP)  const
{
   ELISE_ASSERT(false,"cElNuage3DMaille::ProfOfIndex");
   return 0;
}


double  cElNuage3DMaille::ProfEnPixel(const tIndex2D & aP) const
{
   ELISE_ASSERT(false,"cElNuage3DMaille::ProfEnPixel");
   return 0;
}
double  cElNuage3DMaille::ProfInterpEnPixel(const Pt2dr & aP) const
{
   ELISE_ASSERT(false,"cElNuage3DMaille::ProfInterpEnPixel");
   return 0;
}


Im2D_Bits<1>    cElNuage3DMaille::ImMask() const
{
    return mImDef;
}

Im2DGen *  cElNuage3DMaille::ImProf() const
{
   ELISE_ASSERT(false,"cElNuage3DMaille::ImProf");
   return 0;
}


double cElNuage3DMaille::ProfOfIndexInterpol(const Pt2dr & aPR) const
{
   Pt2di aPI = round_down(aPR);
   double aP1X = aPR.x - aPI.x;
   double aP1Y = aPR.y - aPI.y;
   double aP0X = 1-aP1X;
   double aP0Y = 1-aP1Y;

   return   ProfOfIndex(aPI+Pt2di(0,0))*aP0X*aP0Y
          + ProfOfIndex(aPI+Pt2di(1,0))*aP1X*aP0Y
          + ProfOfIndex(aPI+Pt2di(0,1))*aP0X*aP1Y
          + ProfOfIndex(aPI+Pt2di(1,1))*aP1X*aP1Y ;
}


// Pt2di aPBUG(586,422);


class cBasculeNuage : public cZBuffer
{
     public :

        cBasculeNuage
        (
           cElNuage3DMaille * aDest,
           const cElNuage3DMaille * aInput,
           const cXML_ParamNuage3DMaille * aGeomOutOri
        )  :
           cZBuffer
           (
                 Pt2dr(0,0), Pt2dr(1,1),Pt2dr(0,0),Pt2dr(1,1)
           ),
           mDest  (aDest),
           mInput (aInput),
           mGO    (aGeomOutOri),
           mSzOut (mGO ? mGO->NbPixel() : Pt2di(0,0))
        {
        }

        void Test(const Pt2dr & aP)
        {
           Pt3dr aPEucl = mInput->IndexAndProf2Euclid(Pt2dr(aP.x,aP.y),0);
           Pt3dr aPBasc = mDest->Euclid2ProfAndIndex(aPEucl);

           std::cout << "TEST " << aP << aPEucl  << aPBasc << "\n";
        }

     private :

        Pt3dr ProjTerrain(const Pt3dr & aP) const
        {
           return mDest->Euclid2ProfAndIndex(mInput->IndexAndProf2Euclid(Pt2dr(aP.x,aP.y),aP.z));
        }


        Pt3dr InvProjTerrain(const Pt3dr & aP) const
        {
           Pt3dr aQ = mInput->Euclid2ProfAndIndex(mDest->IndexAndProf2Euclid(Pt2dr(aP.x,aP.y),aP.z));
           return aQ;
           // return mDest->Euclid2ProfAndIndex(mInput->IndexAndProf2Euclid(Pt2dr(aP.x,aP.y),aP.z));
        }

        double ZInterpofXY(const Pt2dr & aP,bool & Ok) const
        {
             Ok = mInput->CaptHasData(aP);
             if (Ok)
                return mInput->ProfOfIndexInterpol(aP);
             return 0.0;
            // �� Pt3dr aPEucl = mInput->IndexAndProf2Euclid(Pt2dr(aP.x,aP.y),0);
            // return  mInput->ProfOfIndex(aP);
        }





        double ZofXY(const Pt2di & aP)   const
        {
              return  mInput->ProfOfIndex(aP);
        }
        bool SelectP(const Pt2di & aP)   const
        {
              return  mInput->IndexHasContenu(aP);
        }

        bool SelectPBascul(const Pt2dr & aP)   const
        {
// std::cout << "SelectPBascul " << mDest << "\n";
// std::cout << "SelectPBascul " << mDest->IndexInsideGeom(round_ni(aP)) << "\n";
// std::cout << "SelectPBascul " << aP << " " << mDest->SzGeom() << " ppp " << mDest->Params().NbPixel() << "\n";
            if (! mGO) return true;
            return (aP.x>0) && (aP.y>0) && (aP.x<mSzOut.x) && (aP.y<mSzOut.y);
        }

        cElNuage3DMaille       * mDest;
        const cElNuage3DMaille * mInput;
        const cXML_ParamNuage3DMaille * mGO;
        Pt2di                            mSzOut;
};

 // cXML_ParamNuage3DMaille aNewParam = CropAndSousEch(mParams,aTr,aScale,aSz);


static    float aBasculeDef = -5e10f;
static    float aBasculeValOut = -4e10f;

cElNuage3DMaille *   cElNuage3DMaille::BasculeInThis
       (
            const cXML_ParamNuage3DMaille * aGeomOutOri,
            const cElNuage3DMaille * aN2,
            bool SupprTriInv,
            double  aCoeffEtire,
            cArgAuxBasc * anAAB,
            cArgAuxBasc_Sec * anAAB2,
            int aLabel,
            bool AutoResize,
            std::vector<Im2DGen *> * aVAttr
       )
{
    AssertNoEmptyData();
    aN2->AssertNoEmptyData();
    if (anAAB)
        SupprTriInv = true;
    cBasculeNuage aBasc(this,aN2,aGeomOutOri);

    if (aVAttr)
    {
        for (int aKA=0 ; aKA<int(aVAttr->size()) ; aKA++)
           aBasc.AddImAttr((*aVAttr)[aKA]);
    }

    if (aCoeffEtire > 0)
        aBasc.InitDynEtirement(aCoeffEtire);
    Pt2di anOfOut;

    bool OkBAII;
    Im2D_REAL4  aMntBasc = aBasc.BasculerAndInterpoleInverse(anOfOut,Pt2di(0,0),aN2->SzUnique(),aBasculeDef,&OkBAII);


    if (!OkBAII)
    {
	std::cout << "Cannot asculerAndInterpoleInverse in cElNuage3DMaille::BasculeInThis";
	//getchar();
        return 0;
    }
    cElNuage3DMaille * aNuageRes = this;


    // Pt2di anOfOutInit= anOfOut;
    if (AutoResize)
    {
         aNuageRes = ReScaleAndClip(Box2dr(anOfOut,anOfOut+aMntBasc.sz()),1.0);
         anOfOut = Pt2di(0,0);
    }


    aNuageRes->FinishBasculeInThis
    (
            aBasc,aMntBasc,anOfOut,
            SupprTriInv,aCoeffEtire,anAAB,anAAB2,aLabel
    );

    if (aVAttr)
    {
       *aVAttr= aBasc.AttrOut();
    }

    return aNuageRes;
}




void   cElNuage3DMaille::FinishBasculeInThis
       (
            cBasculeNuage &  aBasc,
            Im2D_REAL4 aMntBasc,
            Pt2di anOfOut,
            bool SupprTriInv,
            double  aCoeffEtire,
            cArgAuxBasc * anAAB,
            cArgAuxBasc_Sec * anAAB2,
            int aLabel
       )
{
    AssertNoEmptyData();
    TIm2D<float,double>  aTMB(aMntBasc);


    Im2D_Bits<1> aImTI = aBasc.ImTriInv();
    TIm2DBits<1> aTImTI(aImTI);
    // std::cout << anOfOut << "\n";

    Pt2di aP;
    for (aP.x=0 ; aP.x<mSzData.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSzData.y ; aP.y++)
        {
             SetProfOfIndex(aP,0);
             float aV = (float)aTMB.get(aP-anOfOut,aBasculeDef);
             bool aSInv = SupprTriInv  && (aTImTI.get(aP-anOfOut,0) == 1);
             if (anAAB)
             {
                 if (aV>aBasculeValOut)
                 {
                     double aProf1 = ProfOfIndex(aP);
                     if (aV > aProf1)
                     {
                        anAAB2->mImZ.SetR(aP,aProf1);
                        anAAB2->mImInd.SetI(aP,anAAB->mImInd.GetI(aP));
                        anAAB2->mImTriInv.SetI(aP,anAAB->mImTriInv.GetI(aP));

                        SetProfOfIndex(aP,aV);
                        anAAB->mImInd.SetI(aP,aLabel);
                        anAAB->mImTriInv.SetI(aP,aSInv);
                     }
                     else
                     {
                         if (aV>anAAB2->mImZ.GetR(aP))
                         {
                            anAAB2->mImZ.SetR(aP,aV);
                            anAAB2->mImInd.SetI(aP,aLabel);
                            anAAB2->mImTriInv.SetI(aP,aSInv);
                         }
                     }
                 }
             }
             else
             {
                if ((aV<aBasculeValOut) || aSInv)
                {
                   mTImDef.oset(aP,0);
                }
                else
                {
                    mTImDef.oset(aP,1);
                    UpdateDefInterp(aP);
                    SetProfOfIndex(aP,aV);
                }
             }
        }
    }
    if (aCoeffEtire > 0)
    {
      mImEtire = Im2D_U_INT1(mSzData.x,mSzData.y,255);
      Im2D_U_INT1 aIE =  aBasc.ImEtirement();
      ELISE_COPY
      (
          rectangle(anOfOut,anOfOut+aIE.sz()),
          trans(aIE.in(),-anOfOut),
          mImEtire.out()
      );
      // Tiff_Im::CreateFromIm(mImEtire,"Etirement.tif");
    }

}

Im2D_U_INT1  cElNuage3DMaille::ImEtirement()
{
    ELISE_ASSERT
    (
         mImEtire.sz().x > 1,
         "Pas d'image d'etirement calculee !"
    );
    return mImEtire;
}



cElNuage3DMaille *  cElNuage3DMaille::BasculementNewName
                    (
                         const cElNuage3DMaille * aN2,
                         bool SupprTriInv,
                         double aCoeffEtire
                    ) const
{
    cElNuage3DMaille * aRes = Clone();
    aRes->BasculeInThis(0,aN2,SupprTriInv,aCoeffEtire,0,0,-1,false,0);

    return aRes;
}

cArgBacule::cArgBacule(double aSeuilEtir) :
   mSeuilEtir    (aSeuilEtir),
   mDynEtir      (50.0),
   mAutoResize   (true),
   mBoxClipIn    (0),
   mResEtir      (1,1)
{
}
#if (0)
cElNuage3DMaille *  BasculeNuageAutoReSize
                    (
                       const cXML_ParamNuage3DMaille & aGeomOut,
                       const cXML_ParamNuage3DMaille & aGeomIn,
                       const std::string & aDirIn,
                       const std::string &  aNameRes,
                       cArgBacule &
                    )
{
    ELISE_ASSERT(false,"Bascule : service momentanement interrompu pour travaux ;-) ");
    return 0;
}

   bool  AutoResize,
                       const Box2di  * aBoxClipIn,
                       const cArgBacule &    anArg


#endif

void TestNuage(const cElNuage3DMaille * aNu,const std::string & aMes)
{
    int aCpt=0;
    Pt2di aSz = aNu->SzUnique();
    Pt2di aP;
    Im2D_U_INT1 aIm(aSz.x,aSz.y,0);
    TIm2D<U_INT1,INT> aTIm(aIm);
    for ( aP.x=0 ; aP.x<aSz.x ; aP.x++)
    {
        for ( aP.y=0 ; aP.y<aSz.y ; aP.y++)
        {
             if (aNu->IndexHasContenu(aP))
             {
                aTIm.oset(aP,1);
                aCpt++;
             }
        }
    }
    Tiff_Im::CreateFromIm(aIm,aMes+".tif");
    std::cout << "TestNu "  << aMes << " Sz " << aNu->SzUnique() << " NbOk " << aCpt << "\n";
    getchar();
}

cElNuage3DMaille *  BasculeNuageAutoReSize
                    (
                       const cXML_ParamNuage3DMaille & aGeomOutOri,
                       const cXML_ParamNuage3DMaille & aGeomIn,
                       const std::string & aDirIn,
                       const std::string &  aNameRes,
                       cArgBacule &    anArgBasc
                    )
{
   Tiff_Im::SetDefTileFile(100000);
   cXML_ParamNuage3DMaille aGeomOut = aGeomOutOri;

   if (anArgBasc.mAutoResize)
       aGeomOut.NbPixel() = Pt2di(1,1);

   ELISE_ASSERT
   (
       aGeomOutOri.Image_Profondeur().IsInit() && aGeomIn.Image_Profondeur().IsInit(),
      "No prof Im in BasculeNuageAutoReSize"
   );

   std::vector<Im2DGen *> aVAttrIm;
   Im2D_U_INT1 aICor(1,1);
   bool HasCor = false;
   std::string aNameCor;

    aGeomOut.Image_Profondeur().Val().Image() = aGeomIn.Image_Profondeur().Val().Image();
    aGeomOut.Image_Profondeur().Val().Masq() = aGeomIn.Image_Profondeur().Val().Masq();

    if (aGeomIn.Image_Profondeur().Val().Correl().IsInit())
    {
        aNameCor = aDirIn + aGeomIn.Image_Profondeur().Val().Correl().Val();
        Pt2di aSz = Tiff_Im::StdConv(aNameCor).sz();
        Pt2di aP0 (0,0);
        if (anArgBasc.mBoxClipIn)
        {
            aP0 =  anArgBasc.mBoxClipIn->_p0 ;
            aSz =  anArgBasc.mBoxClipIn->sz();
        }


        HasCor = true;

        aICor = Im2D_U_INT1(aSz.x,aSz.y);
        ELISE_COPY(aICor.all_pts(),trans(Tiff_Im::StdConv(aNameCor).in(),aP0),aICor.out());

        //  aICor = Im2D_U_INT1::FromFileStd(aNameCor);
        aVAttrIm.push_back(&aICor);
        aNameCor = NameWithoutDir(aNameRes)+ "_Correl.tif";

        aGeomOut.Image_Profondeur().Val().Correl().SetVal(aNameCor);
    }
    else
    {
       aGeomOut.Image_Profondeur().Val().Correl().SetNoInit();
    }


   cElNuage3DMaille *  aNOut = cElNuage3DMaille::FromParam(aNameRes,aGeomOut,aDirIn,"",1.0,(cParamModifGeomMTDNuage *)0);


   cParamModifGeomMTDNuage * aParamIn = 0;
   if (anArgBasc.mBoxClipIn!=0)
   {
      aParamIn = new cParamModifGeomMTDNuage(1.0,Box2dr(anArgBasc.mBoxClipIn->_p0,anArgBasc.mBoxClipIn->_p1));
   }
   cElNuage3DMaille *  aNIn = cElNuage3DMaille::FromParam(aNameRes,aGeomIn,aDirIn,"",1.0,aParamIn);
   delete aParamIn;


   if (aNIn->IsEmpty())
   {
       return 0;
   }

    cElNuage3DMaille * aRes = aNOut->BasculeInThis(&aGeomOutOri,aNIn,true,anArgBasc.mDynEtir,0,0,-1,anArgBasc.mAutoResize,&aVAttrIm);

   if (! aRes) return 0;



    if (anArgBasc.mDynEtir>0)
    {
       anArgBasc.mResEtir = aRes->ImEtirement();
    }


    if (anArgBasc.mDynEtir>0)
    {
       Im2D_U_INT1 aImEt = aRes->ImEtirement();

       TIm2D<U_INT1,INT> aTImEtir(aImEt);
       Pt2di aSz = aImEt.sz();
       cOptimLabelBinaire * anOLB = cOptimLabelBinaire::ProgDyn(aSz,0.0,1.0);
       // cOptimLabelBinaire * anOLB = cOptimLabelBinaire::CoxRoy(aSz,0.0,1.0);
       Pt2di aP;
       double aSeuil = anArgBasc.mSeuilEtir;
// aSeuil = 0.9;
       //Ewelina : unsed variable commented ou
		//double aDynSeuil = 0.5 / ElMax(aSeuil,1-aSeuil);


       for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
       {
           for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
           {
                 //  aCoefEtire = ElMax(1,ElMin(253,round_ni(mDynEtire/aCoefEtirReel)));
                 double anEtir = anArgBasc.mDynEtir / ElMax(1.0,double(aTImEtir.get(aP)));
                 //  anEtir =  EtirReel
                 anEtir = anEtir / aSeuil;
                 if (anEtir > 1)  anEtir = 2- 1/anEtir;
                 // double aCost = (anEtir-aSeuil)* aDynSeuil;
                 // double aCost = (anEtir*aSeuil)* aDynSeuil;
                 anOLB->SetCost(aP,0.5+ (1-anEtir)/2.0);
                 // anOLB->SetCost(aP,double(aP.x)/aSz.x );
                 // anOLB->SetCost(aP,1);
           }
       }
       Im2D_Bits<1> aSol = anOLB->Sol();
       int aOk;
       ELISE_COPY(aSol.all_pts(),aSol.in(),sigma(aOk));


       ELISE_COPY
       (
          aRes->ImDef().all_pts(),
          (aRes->ImDef().in() && aSol.in()),
          aRes->ImDef().out()
       );

       delete anOLB;
    }
    if (HasCor)
    {
        if (1)
        {
            Tiff_Im::Create8BFromFonc
            (
                aDirIn+ aNameCor,
                aVAttrIm[0]->sz(),
                aVAttrIm[0]->in()
            );
        }
        else
        {
            ELISE_ASSERT(aVAttrIm[0]->sz()==aRes->ImEtirement().sz(),"Etir an cor : sz dif ??\n");
            // ELISE_COPY(aRes->ImEtirement().all_pts(),FX%256,aRes->ImEtirement().out());
            Tiff_Im::Create8BFromFonc
            (
                aDirIn+ aNameCor,
                aRes->ImEtirement().sz(),
                aRes->ImEtirement().in()
            );
        }
    }


   return aRes;
}
/*
*/

double Resol(const cXML_ParamNuage3DMaille & aNuage)
{
   ElAffin2D  aM2C =    Xml2EL(aNuage.Orientation().OrIntImaM2C()); // RPCNuage
   ElAffin2D aC2M = aM2C.inv();
   return (euclid(aC2M.I10()) + euclid(aC2M.I01()))/2.0;
}

double DynProfInPixel(const cXML_ParamNuage3DMaille & aNuage)
{
   double aSzPixel = Resol(aNuage);

   return (aSzPixel * aNuage.RatioResolAltiPlani().Val()) / (aNuage.Image_Profondeur().Val().ResolutionAlti());

}

double cElNuage3DMaille::DynProfInPixel() const
{
   return ::DynProfInPixel(Params());
}


double cElNuage3DMaille::SeuilDistPbTopo() const
{
   if (mAnam) 
   {
      return mAnam->SeuilDistPbTopo() / Resol(Params());
   }
   return 0;
}


// Pour teni compt du fait que la resolution a pu etre multipliee,
// Ex 0.12 initiale , on finit a DeZoom 2, donc devienr 0.24 qui n'est pas un chiffre rond
// (on peut avoir 23, 48, 96, 192  .... comme chifre rond)

bool ToFOMMulResolStdRound(double & aVal0,int aMul)
{
    double aVDiv = aVal0 / aMul;

    cDecimal aDec = StdRound(aVDiv);
    double aNewV = aDec.RVal();
    double aDif = ElAbs(aNewV-aVDiv)/(ElAbs(aVDiv)) ;

    if (aDif<1e-5)
    {
        aVal0 = aNewV * aMul;
        return true;
    }
    return false;
}

bool ToFOMMulResolStdRound(double & aVal0)
{
   for (int aP=1 ; aP<64 ; aP*=2)
   {
       if (ToFOMMulResolStdRound(aVal0,aP))
           return true;
   }
   return false;
}

void ToFOMResolStdRound(double & aVal)
{
   if (ElAbs(aVal) < 1e-20) return;
   
   int aSign=1;
   if (aVal<0)
   {
        aVal =-aVal;
        aSign=-1;
   }

    
   if ( ! ToFOMMulResolStdRound(aVal))
   {
       std::cout << "ToFOMResolStdRound:: " << aVal  << "\n";
       ELISE_ASSERT(false,"RESOL ToFOMStdRound");
   }

   aVal *=  aSign;
}

void ToFOMOriStdRound(double & aVal,const double & aResol)
{
    double aRatio = aVal / aResol;

//	cDecimal aDec = StdRound(aRatio);
//    double aIR = aDec.RVal();
    double aIR = round_ni(aRatio);
    double aDif = ElAbs(aRatio-aIR);
    // MPD : 01/06/2015 , Brutal-in�legant-dangdereux mais efficace
    // L'origine du pb vient d'arrondi avec les "grande" coordonnes (type lambert) et petite
    // resol (genre < au mm); Bon ce serait + propre de gere des calcul exact ou au moins precis dans les XML
    // mais sans soute toute une chaine a remonter ...
    if (aDif>= 1e-7)
    {
       std::cout << "Waring supciciousrs round, DIF " << aDif  << " at " << __LINE__ << " of " << __FILE__ << "\n";
       if (aDif>= 1e-2)
          ELISE_ASSERT(false,"ORI ToFOMStdRound");
    }

    aVal = aResol * aIR;
}


cFileOriMnt ToFOM(const cXML_ParamNuage3DMaille & aXML,bool StdRound)
{
    cFileOriMnt aRes;
    ELISE_ASSERT(aXML.Image_Profondeur().IsInit(),"ToFOM => Image_Profondeur");
    const cImage_Profondeur & anIP= aXML.Image_Profondeur().Val();
    aRes.NameFileMnt() = anIP.Image();
    aRes.NameFileMasque() = anIP.Masq();
    aRes.NombrePixels() = aXML.NbPixel();
    // aRes.NameFileMnt() = 

    double anOriA = anIP.OrigineAlti();
    double aResA = anIP.ResolutionAlti();

    ElAffin2D  anAff = Xml2EL(aXML.Orientation().OrIntImaM2C()); // RPCNuage
    anAff = anAff.inv();

    Pt2dr anOriPlani = anAff.I00();
    // std::cout << "ORIPLANI " << anOriPlani << "\n";
    // Pt2dr 
    Pt2dr aResolPlani(anAff.I10().x,anAff.I01().y);
    // std::cout << "RESOL LANI " << aResolPlani << "\n";

    double anErr = (ElAbs(anAff.I10().y) + ElAbs(anAff.I01().x)) / euclid(aResolPlani);
    ELISE_ASSERT(anErr<1e-7,"ToFOM Affinite non inv");
    

    if (StdRound)
    {
        ToFOMResolStdRound(aResolPlani.x);
        ToFOMResolStdRound(aResolPlani.y);
        ToFOMResolStdRound(aResA);

        ToFOMOriStdRound(anOriPlani.x,aResolPlani.x);
        ToFOMOriStdRound(anOriPlani.y,aResolPlani.y);
        ToFOMOriStdRound(anOriA,aResA);
    }

    aRes.OriginePlani() = anOriPlani;
    aRes.ResolutionPlani() = aResolPlani;
    aRes.OrigineAlti() = anOriA;
    aRes.ResolutionAlti() = aResA;
    aRes.Geometrie() = anIP.GeomRestit();

    return aRes;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
