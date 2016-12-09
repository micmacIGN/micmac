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


/*
   Sur le long terme, modifier :

     - la possiblite d'avoir des fonctions multiple (Dim N)
     - associer a un fctr, un cSetEqFormelles
*/

#include "StdAfx.h"


/****************************************/
/*                                      */
/*           cEqVueLaserImage           */
/*                                      */
/****************************************/

// aRotPts Si non nulle, il faut multiplier les points par aRotPts
//  pour les avoir en "vrai terrain"


cEqVueLaserImage::cEqVueLaserImage
(
      cRotationFormelle *aRotPts,
      bool Multi,
      bool Normalize,
      INT aNbPts,
      cLIParam_Image & anImA,
      cLIParam_Image & anImB,
      bool GenCode
)  :
   mSet     (anImA.Set()),
   mMakA    ("_A_",anImA,Normalize,aNbPts,aRotPts),
   mMakB    ("_B_",anImB,Normalize,aNbPts,aRotPts),
   mRotPts  (aRotPts),
   mName     (     std::string("cEqCorLI_") 
                  + std::string(Multi ? "Multi_" : "Single_")
                  + ToString(aNbPts)
                  + (aRotPts ? (std::string("_DRPts")+ToString(aRotPts->Degre())): "")
                  + std::string(Normalize ? "" : "_NonNorm")
	       ),
   mEq         (mMakA.FRad() - mMakB.FRad()),
   mMulti      (Multi),
   mAdrTimeRot (0),
   mNormalize  (Normalize)
{

     
     ELISE_ASSERT(&(anImA.Set())==&(anImB.Set()),"Diff Eq in cEqVueLaserImage");
     anImA.Rot().IncInterv().SetName("OrA");
     anImB.Rot().IncInterv().SetName("OrB");


     mLInterv.AddInterv(anImA.Rot().IncInterv());
     mLInterv.AddInterv(anImB.Rot().IncInterv());

     if (aRotPts)
     {
        aRotPts->IncInterv().SetName("OrPts");
        mLInterv.AddInterv(aRotPts->IncInterv());
     }

     mFoncEq = cElCompiledFonc::AllocFromName(mName);

     if (GenCode)
     {
	   cout << "FCTR = " << mFoncEq << "\n";
	   INT Sz = mMulti ? aNbPts : 1;
	   std::vector<Fonc_Num> vEq;
	   for (INT aK=0 ; aK< Sz ; aK++)
               vEq.push_back(mMakA.KthRad(aK)-mMakB.KthRad(aK));
            cElCompileFN::DoEverything
            (
                    "src/GC_photogram/",
                    mName,
                    vEq,
                    mLInterv
             );
	    return;

     }
     ELISE_ASSERT(mFoncEq!=0,"Cannot Find Fctr in cEqVueLaserImage::cEqVueLaserImage");

     mFoncEq->SetMappingCur(mLInterv,&mSet);
     mMakA.InitAdr(mFoncEq);
     mMakB.InitAdr(mFoncEq);
     if (aRotPts && (aRotPts->Degre()>0))
        mAdrTimeRot = mFoncEq->RequireAdrVarLocFromString(aRotPts->NameParamTime());


     // RequireAdrVarLocFromString
     mSet.AddFonct(mFoncEq);
}

void cEqVueLaserImage::Update_0F2D()
{
}

/*
ElRotation3D cEqVueLaserImage::CurRotPtsOfT(REAL aT)
{
    return  mRotPts                           ?
            mRotPts->CurRot(aT)               :
            ElRotation3D(Pt3dr(0,0,0),0,0,0)  ;
}
*/

REAL  cEqVueLaserImage::AddAllEquations(std::vector<Pt3dr> Pts,REAL aPds,REAL aTime)
{
     if (mAdrTimeRot)
        *mAdrTimeRot = aTime;
     INT aNb = mMulti ? 1 : INT(Pts.size());
     INT Cpt = 0;

     ElRotation3D aRotOfT =  mRotPts                           ?
                             mRotPts->CurRot(aTime)            :
                             ElRotation3D(Pt3dr(0,0,0),0,0,0)  ;

     for (INT aK=0 ; aK<aNb ; aK++)
     {
         if (   mMakA.InitEquations(Pts,aK,aRotOfT)
             && mMakB.InitEquations(Pts,aK,aRotOfT)
	    )
	 {
	    Cpt++;
	 }
     }
     if (Cpt != aNb)
	return -1;

     Cpt = 0;
     REAL aRes = 0.0;
     for (INT aK=0 ; aK<aNb ; aK++)
     {
         if (   mMakA.InitEquations(Pts,aK,aRotOfT)
             && mMakB.InitEquations(Pts,aK,aRotOfT)
	    )
	 {
             const std::vector<REAL> & vVal =  
		     mSet.VAddEqFonctToSys(mFoncEq,aPds,false,NullPCVU);
	     for (INT aK=0 ; aK<INT(vVal.size()) ; aK++)
	     {
	         aRes += ElSquare(vVal[aK]);
		 Cpt++;
	     }
	 }
     }
     return aRes / Cpt;
}

/****************************************/
/*                                      */
/*             cLIParam_Image           */
/*                                      */
/****************************************/

cRotationFormelle & cLIParam_Image::Rot() {return *pRot;}
cSetEqFormelles & cLIParam_Image::Set() {return mSet;}
CamStenope & cLIParam_Image::Cam() {return mCam;}
Im2D_REAL4 cLIParam_Image::Im() {return mIm;}

void cLIParam_Image::UpdateCam()
{
	mCam.SetOrientation(pRot->CurRot().inv());
}
void cLIParam_Image::Update_0F2D()
{
     UpdateCam();
}

cLIParam_Image::cLIParam_Image
(
        cSetEqFormelles & aSet,
        Im2D_REAL4   anIm,
        REAL         aZoom,
        CamStenope & aCam,
        cNameSpaceEqF::eModeContrRot aMode
) :
   mSet   (aSet),
   pRot   (mSet.NewRotation(aMode,aCam.Orient().inv())),
   mIm    (anIm),
   mZoom  (aZoom),
   mCam   (aCam)
{
}

void cLIParam_Image::SetImZ(Im2D_REAL4 anIm,REAL aZ)
{
    mIm =  anIm;
    mZoom = aZ;
}


Pt2dr  cLIParam_Image::Ray2Im(Pt3dr aP)
{
     return mCam.R3toF2(aP) / mZoom;
}

Pt3dr cLIParam_Image::Im2Ray(Pt2dr aP)
{
   return mCam.F2toDirRayonL3(aP * mZoom);
}



/****************************************/
/*                                      */
/*             cLI_MakeEqIm             */
/*                                      */
/****************************************/

cLI_MakeEqIm::cLI_MakeEqIm
(
     const std::string & aPref,
     cLIParam_Image         & anIm,
     bool       Normalize,
     INT        aNbPts,
     cRotationFormelle * aRotPts
) :
   mPref  (aPref),
   mIm    (anIm),
   mRot   (anIm.Rot()),
   mIncPts   (),
   mMoy   (0),
   mEct   (0),
   mMat   (3,3),
   mNormalize (Normalize)
{
    
    for (INT aK=0 ; aK<aNbPts; aK++)
    {
        mIncPts.push_back(new cPts(mPref,aK,mRot,aRotPts));
	mMoy = mMoy+mIncPts.back()->mRad;
	mEct = mEct+Square(mIncPts.back()->mRad);
    }

    mMoy  = mMoy/aNbPts;
    mEct = sqrt(mEct/aNbPts-Square(mMoy));

    // mRadC =  (mIncPts.front()->mRad-mMoy) / mEct;
    mRadC = KthRad(0);
}
Fonc_Num cLI_MakeEqIm::KthRad(INT aK)
{
    return  mNormalize                       ?
            (mIncPts[aK]->mRad-mMoy) / mEct  :
            mIncPts[aK]->mRad                ;

}

void cLI_MakeEqIm::InitAdr(cElCompiledFonc * aFctr)
{
     for (INT aK=0 ; aK<INT(mIncPts.size()) ; aK++)
         mIncPts[aK]->InitAdr(aFctr);
}

Fonc_Num    cLI_MakeEqIm::FRad()
{
     return mRadC;
}

Fonc_Num cLI_MakeEqIm::cPts::RadiomOfP(Pt3d<Fonc_Num> aPM)
{
     if (mRotPts)
        aPM = mRotPts->C2M(aPM);
     Pt3d<Fonc_Num> aPCam = mRotIm.M2C(aPM);
     Fonc_Num U = aPCam.x / aPCam.z;
     Fonc_Num V = aPCam.y / aPCam.z;
     return  m0 + mU*U + mV *V;
}


   
bool  cLI_MakeEqIm::InitEquations(std::vector<Pt3dr> mP3D,INT Offset,const ElRotation3D & aRot)
{

     ELISE_ASSERT
     (
	     mIncPts.size()  == mP3D.size(),
	     "Bas Size in cLI_MakeEqIm::InitEquations"
     );

     for (INT aK =0 ; aK<INT(mIncPts.size()) ; aK++)
	  if (! InitOneEquation(mP3D[(aK+Offset)%mP3D.size()],*mIncPts[aK],aRot))
             return false;
    return true;
}

bool cLI_MakeEqIm::InitOneEquation(Pt3dr aP3,cPts & anInc,const ElRotation3D & aRot)
{
    aP3 = aRot.ImAff(aP3);
    TIm2D<REAL4,REAL8>    aTIm(mIm.Im());
    Pt2dr aP2 = mIm.Ray2Im(aP3);
    if (! aTIm.inside_rab(aP2,3))
       return false;
    Pt2di aQ0  = round_down(aP2);
    Pt2di aQ1 = aQ0 + Pt2di(1,0);
    Pt2di aQ2 = aQ0 + Pt2di(0,1);
    Pt2dr aRes = aP2 - Pt2dr(aQ0);
    if (aRes.x+aRes.y > 1.0)
	aQ0 += Pt2di(1,1);

    REAL RO = aTIm.get(aQ0);
    REAL R1 = aTIm.get(aQ1);
    REAL R2 = aTIm.get(aQ2);

    Pt3dr UV0 = mIm.Im2Ray(Pt2dr(aQ0));
    Pt3dr UV1 = mIm.Im2Ray(Pt2dr(aQ1));
    Pt3dr UV2 = mIm.Im2Ray(Pt2dr(aQ2));

    SetLig(mMat,0,UV0);
    SetLig(mMat,1,UV1);
    SetLig(mMat,2,UV2);

    if (! self_gaussj_svp(mMat))
	    return false;

    Pt3dr aSol = mMat * Pt3dr(RO,R1,R2);


    *(anInc.mAdrX) = aP3.x;
    *(anInc.mAdrY) = aP3.y;
    *(anInc.mAdrZ) = aP3.z;

    *(anInc.mAdrU) = aSol.x;
    *(anInc.mAdrV) = aSol.y;
    *(anInc.mAdr0) = aSol.z;

    return true;
}

	      //  -----------------
	      //  cLI_MakeEqIm::cPts
	      //  -----------------



cLI_MakeEqIm::cPts::cPts
(
         const std::string & aPref,
         INT aNum,
         cRotationFormelle & aRot,
         cRotationFormelle * aRotPts
) :
    mRotIm  (aRot),
    mRotPts (aRotPts),
    m0      (0.0,std::string("Cste")+aPref+ToString(aNum)),
    mU      (0.0,std::string("CoeffU")+aPref+ ToString(aNum)),
    mV      (0.0,std::string("CoeffV")+aPref+ ToString(aNum)),
    mX      (0.0,std::string("P3X")+aPref+ToString(aNum)),
    mY      (0.0,std::string("P3Y")+aPref+ToString(aNum)),
    mZ      (0.0,std::string("P3Z")+aPref+ToString(aNum)),
    mRad    (RadiomOfP(Pt3d<Fonc_Num>(mX,mY,mZ)))
{
}

void cLI_MakeEqIm::cPts::InitAdr(cElCompiledFonc * aFctr)
{
    mAdr0 = aFctr->RequireAdrVarLocFromString(m0.Name());
    mAdrU = aFctr->RequireAdrVarLocFromString(mU.Name());
    mAdrV = aFctr->RequireAdrVarLocFromString(mV.Name());
    mAdrX = aFctr->RequireAdrVarLocFromString(mX.Name());
    mAdrY = aFctr->RequireAdrVarLocFromString(mY.Name());
    mAdrZ = aFctr->RequireAdrVarLocFromString(mZ.Name());
}

/****************************************/
/*                                      */
/*                ::                    */
/*                                      */
/****************************************/

/*
void GenCodeLaserImage(INT aNb)
{
     CamStenopeIdeale aCam(1.0,Pt2dr(0,0));
     cSetEqFormelles aSet;
     Im2D_REAL4 aI(1,1);

      cLIParam_Image  * aP1 =  aSet.NewLIParamImage(aI,1.0,aCam);
      cLIParam_Image  * aP2 =  aSet.NewLIParamImage(aI,1.0,aCam);

      aSet.NewLIEqVueLaserIm(aNb,*aP1,*aP2);
      ElEXIT(-1);
}
*/

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
