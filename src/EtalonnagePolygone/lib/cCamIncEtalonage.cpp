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
*/

#include "StdAfx.h"


/************************************************************/
/*                                                          */
/*                   cCamIncEtalonage                       */
/*                                                          */
/************************************************************/

static std::vector<double> NoParAdd;

cCamStenopeDistRadPol * cCamIncEtalonage::CurCam() const
{
   cCamStenopeDistRadPol *  aCam = 
	   new cCamStenopeDistRadPol
                         (
			    mEtal.Param().ModeC2M(),
                            mEtal.CurFoc(),
                            mEtal.CurPP(),
                            mEtal.CurDist(),
                            NoParAdd
                         );

    aCam->SetOrientation(pCF->CurRot().inv());
    return aCam;
}

CamStenope  * cCamIncEtalonage::CurCamGen() const
{
   CamStenope *  aCam =  pCF->PIF().CurPIF();
   aCam->SetOrientation(pCF->CurRot().inv());
   return aCam;
}





void cCamIncEtalonage::SauvEOR()
{
    cCamStenopeDistRadPol * pCam = CurCam();
    cout << "For " << mName << " Copt = " << pCam->VraiOpticalCenter() << "\n";
    std::string aName = mEtal.Param().Directory() + mName + ".eor";
    ELISE_fp aFile(aName.c_str(),ELISE_fp::WRITE);
    pCam->write(aFile);
    aFile.close();
    delete pCam;
}

cSetPointes1Im * cCamIncEtalonage::PointeInitial() const
{
   if (mPointeInitial==0)
   {
      mPointeInitial = new cSetPointes1Im
                           (
			       mEtal.Polygone(),
			       mEtal.NamePointeInit(mName),
			       false
			   );
   }
  
   return mPointeInitial;
}

cCamIncEtalonage::cCamIncEtalonage
(
      INT                 aNumCam,
      cEtalonnage &       anEtal,
      const std::string & aNameTiff,
      const std::string & aShortName,
      bool                PointeCanBeVide,
      const std::string & aNamePointes
)  :
    mNumCam (aNumCam),
    mEtal   (anEtal), 
    mPointes(mEtal.Polygone(),aNamePointes,PointeCanBeVide),
    mNamePointes (aNamePointes),
    mName   (aShortName),
    mNameTiff  (),
    pCF     (0),
    mCam    (anEtal.Param().ModeC2M(),anEtal.NormFoc0(),anEtal.NormPP0(),NoParAdd),
    mTiff   (aNameTiff),
    mPointeInitial (0),
    mUseDirectPointeManuel (false)
{
   std::string aDir;
   SplitDirAndFile(aDir,mNameTiff,aNameTiff);

   const cImageUseDirectPointeManuel * aIU = anEtal.Param().ImageUseDirectPointeManuel().PtrVal();
   if (aIU)
   {
      const std::list<cElRegex_Ptr> & aL = aIU->Id();

      for(std::list<cElRegex_Ptr>::const_iterator iTP=aL.begin(); iTP!=aL.end() ; iTP++)
      {
         if ((*iTP)->Match(mName))
            mUseDirectPointeManuel = true;
      }
   }


    tLPointes & aL = mPointes.Pointes();
    for
    (
          tLPointes::iterator iT = aL.begin();
          iT != aL.end();
          iT++
    )
    {
         if (anEtal.Param().InvYPointe())
         {
             iT->SetPosIm(iT->PosIm().conj());
         }
	 Appar23 anAp(iT->PosIm(),iT->PosTer());
	 //  anAp.SetNum(iT->Cible().Ind());
	 mAppuisInit.push_back(anAp);
	 mIndAppuisInit.push_back(iT->Cible().Ind());
	 // std::cout << "IND " << iT->Cible().Ind() << "\n";
	 // mAppuisInit.push_back(Appar23(iT->PosIm(),iT->PosTer()));
         iT->SetPosIm(mEtal.ToPN(iT->PosIm()));
    }

}

void cCamIncEtalonage::ExportAsDico(const cExportAppuisAsDico& anExp)
{
    cSetOfMesureAppuisFlottants  aSet;
    aSet.MesureAppuiFlottant1Im().push_back(cMesureAppuiFlottant1Im());
    cMesureAppuiFlottant1Im & aMAF = aSet.MesureAppuiFlottant1Im().back();

    aMAF.NameIm() = mNameTiff;
    // aMAF.NameDico() = anExp.NameDico();

    std::list<int>::const_iterator itI = mIndAppuisInit.begin();
    for
    (
       std::list<Appar23>::const_iterator itA = mAppuisInit.begin() ;
       itA != mAppuisInit.end() ;
       itA++,itI++
    )
    {
        cOneMesureAF1I aM;
	aM.NamePt() = ToString(*itI);
	aM.PtIm() = itA->pim;

	aMAF.OneMesureAF1I().push_back(aM);
    }

    MakeFileXML
    (
       aSet,
         mEtal.Param().Directory() 
       + std::string("AppOnDico_") + anExp.NameDico()
       + std::string("_")+ mName  +std::string(".xml")
    );
}


bool cCamIncEtalonage::UseDirectPointeManuel() const
{
   return mUseDirectPointeManuel;
}


const std::list<Appar23> & cCamIncEtalonage::AppuisInit() const
{
  return mAppuisInit;
}

const std::list<int> & cCamIncEtalonage::IndAppuisInit() const
{
  return mIndAppuisInit;
}

const cSetPointes1Im & cCamIncEtalonage::Pointes()  const
{
   return mPointes;
}

const std::string & cCamIncEtalonage::NamePointes() const
{
   return mNamePointes;
}



const std::string & cCamIncEtalonage::Name() const
{
    return mName;
}


const CamStenopeIdeale & cCamIncEtalonage::Cam() const
{
   return mCam;
}


cSetPointes1Im &  cCamIncEtalonage::SetPointes()
{
     return mPointes;
}


cCameraFormelle & cCamIncEtalonage::CF()
{
     return *pCF;
}

Tiff_Im cCamIncEtalonage::Tiff() const
{
    return mTiff.ImGray8B();
}

Pt2dr cCamIncEtalonage::Terrain2ImageGlob(Pt3dr aP) const
{
      return mEtal.FromPN(mCam.R3toF2(aP));
}


void cCamIncEtalonage::TestOrient(ElRotation3D aRot)
{
     REAL SomD= 0.0;
     mCam.SetOrientation(aRot);
     std::list<Appar23> ApInit = StdAppuis(true);
     for 
     (
         std::list<Appar23>::iterator it23=ApInit.begin();
         it23 != ApInit.end() ;
         it23++
     )
     {
        REAL aD = euclid(it23->pim,mCam.R3toF2(it23->pter));
	cout << aD << mCam.R3toL3(it23->pter) << "\n";
	SomD += aD;
     }
     cout << "SomD = " << SomD << "\n";
     cout << "\n";
}

int NB_RANSAC = 500;

void cCamIncEtalonage::TestOrient()
{
   std::list<ElRotation3D> aList;
   mCam.OrientFromPtsAppui(aList,StdAppuis(true));

   for 
   (
        std::list<ElRotation3D>::iterator itR = aList.begin();
	itR != aList.end();
	itR++
   )
   {
     TestOrient(*itR);
   }
   REAL DMin;
   ElRotation3D aRot = mCam.RansacOFPA(true,NB_RANSAC,StdAppuis(true),&DMin);
   cout << "----------- COMBINATOIRE -------- \n";
   TestOrient(aRot);
}


ElRotation3D cCamIncEtalonage::RotationInitiale()
{
    REAL DMin;
    std::list<Appar23> ApInit = StdAppuis(true);

    if (ApInit.size() >= 4)
    { 
       std::cout << "OFPA : Begin \n";
       // int aNb = mEtal.Param().NbPMaxOrient();
       // std::cout << "NB=" << aNb << "\n"; getchar();
       ElRotation3D aRot = mCam.RansacOFPA(true,NB_RANSAC,ApInit,&DMin);
       std::cout << "NbP = " << ApInit.size() << "\n";
       std::cout << "OFPA : End ; D=" << DMin  << "\n";
       return aRot;
    }


    cEtalonnage &  E2 = mEtal.EtalRatt();

    std::string NR = mEtal.Param().NamePosRatt();

    ElRotation3D A2 = mEtal.GetBestRotEstim(NR);
    ElRotation3D B1 = E2.GetBestRotEstim(mName);
    ElRotation3D B2 = E2.GetBestRotEstim(NR);

    return A2 * B2.inv() * B1;
    // ELISE_ASSERT(false,"Cannot Initialize rotation");
}


void cCamIncEtalonage::InitOrient
     (cParamIntrinsequeFormel * aPIF, cCamIncEtalonage * pCamRatt)
{

    ElRotation3D aRot  = RotationInitiale();

     cNameSpaceEqF::eModeContrRot aMode = cNameSpaceEqF::eRotLibre;
     if (mNumCam==0)
        aMode = cNameSpaceEqF::eRotFigee;
     else if (mNumCam==1)
        aMode = cNameSpaceEqF::eRotBaseU;

     pCF = aPIF->NewCam
           (
                aMode,
                aRot.inv(),
                pCamRatt ? pCamRatt->pCF : 0,
                mName,
                true
           );
     mCam.SetOrientation(aRot);
}


std::list<Appar23> cCamIncEtalonage::StdAppuis(bool doPImNorm)
{
	return mEtal.StdAppuis(doPImNorm,mName,&mPointes);
}


    //   =================================================
    //    cEtalonnage
    //   =================================================


std::list<Appar23> cEtalonnage::StdAppuis(bool doPImNorm,std::string & aStr,cSetPointes1Im * SetPrecis)
{
    std::list<Appar23> aRes;
    cSetPointes1Im aSet4(mPol,NamePointeInit(aStr),true);
    tLPointes & aL4 = aSet4.Pointes();
    for (tLPointes::iterator iT=aL4.begin(); iT!=aL4.end(); iT++)
    {
	INT Ind = iT->Cible().Ind();
	Pt3dr PTer = iT->PosTer();
	Pt2dr PIm = iT->PosIm();
	if (doPImNorm)
	   PIm = ToPN(PIm);

	if (SetPrecis)
	{
            cPointeEtalonage * aPointe = SetPrecis->PointeOfIdSvp(Ind);
	    if (aPointe)
	    {
		ELISE_ASSERT(euclid(PTer-aPointe->PosTer())<1e-8,"cEtalonnage::StdAppuis !");
                PTer = aPointe->PosTer(); // Devrait Rien changer !!
                PIm = aPointe->PosIm();
	    }
	}

        aRes.push_back(Appar23(PIm,PTer));
    }
    return aRes;
}

ElRotation3D cEtalonnage::RotationFromAppui(std::string & aStr,cSetPointes1Im * aSetP)
{
     std::list<Appar23> aLApp = StdAppuis(true,aStr,aSetP);
     CamStenopeIdeale aCam(mParam.ModeC2M(),1.0,Pt2dr(0,0),NoParAdd);
     REAL DMin;
     ElRotation3D aRes= aCam.RansacOFPA(true,NB_RANSAC,aLApp,&DMin);
     return aRes;
}


ElRotation3D cEtalonnage::GetBestRotEstim(const std::string & aNameCam)
{
     for (INT aK=0 ; aK<NbRotEstim ; aK++)
     {
         std::string  aName = *(TheRotPoss[aK]);
	 aName = NameRot(aName,aNameCam);
         if (ELISE_fp::exist_file(aName.c_str()))
	 {
            return ReadFromFile((ElRotation3D *)0,aName);
	 }
     }

     for (INT aK=0 ; aK<NbRotEstim ; aK++)
     {
         std::string  aName = *(TheRotPoss[aK]);
	 aName = NameRot(aName,aNameCam);
         std::cout << "TESTED " << aName << "\n";
     }




     cout << "FOR CAM : " << aNameCam << "\n";
     ELISE_ASSERT(false,"cEtalonnage::GetBestRotEstim");
     return ElRotation3D(Pt3dr(0,0,0),0,0,0);
}




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
