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

#include "NewOri.h"




const std::string ExtTxtXml = "xml";
const std::string ExtBinDmp = "dmp";
const std::string & ExtXml(bool Bin)
{
   return Bin ? ExtBinDmp : ExtTxtXml;
}


const std::string  cNewO_NameManager::PrefixDirTmp = "NewOriTmp";


cNewO_NameManager::cNewO_NameManager
(
     const std::string  & aExtName,
     const std::string  & aPrefHom,
     bool                 Quick,
     const std::string  & aDir,
     const std::string  & anOriCal,
     const std::string  & aPostHom,
     const std::string  & aOriOut
) :
    mICNM        (cInterfChantierNameManipulateur::BasicAlloc(aDir)),
    mDir         (aDir),
    mPrefOriCal  (anOriCal),
    mPostHom     (aPostHom),
    mPrefHom     (aPrefHom),
    mExtName     (aExtName),
    mQuick       (Quick),
    mOriOut      (aOriOut)
{
   if (mOriOut=="") 
      mOriOut = (mQuick ? "Martini" : "MartiniGin") + mExtName +mPrefHom + anOriCal;

   StdCorrecNameOrient(mPrefOriCal,mDir);
   mPostfixDir    =   mExtName + mPrefHom +  mPrefOriCal + std::string(mQuick ? "Quick" : "Std");
   mDirTmp      =   std::string(PrefixDirTmp) + mPostfixDir + "/";

   ELISE_fp::MkDir(mDir+"Ori-"+mOriOut+"/");
}


cVirtInterf_NewO_NameManager * cVirtInterf_NewO_NameManager::StdAlloc(const std::string & aDir, const std::string  & anOri, bool  Quick)
{

   return new cNewO_NameManager("","",Quick,aDir,anOri,"dat");
}

//=============== Surcharge method
//           "cVirtInterf_NewO_NameManager * cVirtInterf_NewO_NameManager::StdAlloc"
//                 pour adapter avec Suffix homol

cVirtInterf_NewO_NameManager * cVirtInterf_NewO_NameManager::StdAlloc(const std::string  & aPrefHom, const std::string & aDir,const std::string  & anOri,bool  Quick)
{

   return new cNewO_NameManager("",aPrefHom,Quick,aDir,anOri,"dat");
}
//=====================================================================================================

const std::string & cNewO_NameManager::Dir() const
{
   return mDir;
}

const std::string & cNewO_NameManager::OriOut() const
{
   return mOriOut;
}


std::string cNewO_NameManager::KeySetCpleOri() const
{
   return "NKS-Set-NewOri-CplIm2OriRel@"+mPostfixDir +"@dmp";
}

std::string cNewO_NameManager::KeyAssocCpleOri() const
{
   return "NKS-Assoc-NewOri-CplIm2OriRel@"+mPostfixDir +"@dmp";
}




ElPackHomologue cNewO_NameManager::PackOfName(const std::string & aN1,const std::string & aN2) const
{
    std::string aNameH = mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mPrefHom+"@"+mPostHom,aN1,aN2,true);
    if (!  ELISE_fp::exist_file(aNameH))
       return ElPackHomologue();

    return ElPackHomologue::FromFile(aNameH);
}

cInterfChantierNameManipulateur *  cNewO_NameManager::ICNM()
{
   return mICNM;
}


CamStenope * cInterfChantierNameManipulateur::GlobCalibOfName(const std::string  & aName,const std::string & aPrefOriCal,bool aModeFraser) 
{
   // std::cout << "cInterfChantierNameManipulateur::GlobCalibOfName " << aPrefOriCal << "\n"; getchar();


   if (aPrefOriCal =="")
   {
        cMetaDataPhoto aMTD = cMetaDataPhoto::CreateExiv2(Dir() +aName);
        std::vector<double> aPAF;
        double aFPix  = aMTD.FocPix();
        Pt2di  aSzIm  = aMTD.XifSzIm();
        Pt2dr  aPP = Pt2dr(aSzIm) / 2.0;

        bool IsFE;
        FromString(IsFE,Assoc1To1("NKS-IsFishEye",aName,true));
        std::string aNameCal = "CamF" + ToString(aFPix) +"_Sz"+ToString(aSzIm) + "FE"+ToString(IsFE);
        if (DicBoolFind(mMapName2Calib,aNameCal))
           return mMapName2Calib[aNameCal];
        CamStenope * aRes = 0;

        std::vector<double> aVP;
        std::vector<double> aVE;
        if (IsFE)
        {
            aVE.push_back(aFPix);
            aVP.push_back(aPP.x);
            aVP.push_back(aPP.y);
            aRes = new cCamLin_FishEye_10_5_5
                       (
                            false,
                            aFPix,aPP,Pt2dr(aSzIm),
                            aPAF,
                            &aVP,
                            &aVE
                       );

        }
        else
        {
// std::cout << "aModeFraseraModeFraser " << aModeFraser << "\n"; getchar();
             if (aModeFraser)
                aRes = new cCam_Fraser_PPaEqPPs(false,aFPix,aPP,Pt2dr(aSzIm),aPAF,&aVP,&aVE);
             else
                aRes = new CamStenopeIdeale(false,aFPix,aPP,aPAF);
        }
        aRes->SetSz(aSzIm);
        mMapName2Calib[aNameCal] =  aRes;
        return aRes;
   }

   std::string  aNC = StdNameCalib(aPrefOriCal,aName);

   if (DicBoolFind(mMapName2Calib,aNC))
      return mMapName2Calib[aNC];

   mMapName2Calib[aNC] =  CamOrientGenFromFile(aNC,this);

   return mMapName2Calib[aNC];
}

CamStenope * cNewO_NameManager::CamOfName(const std::string  & aName)  const
{
    return mICNM->GlobCalibOfName(aName,mPrefOriCal,true);
}

CamStenope * cNewO_NameManager::CalibrationCamera(const std::string  & aName) const
{
   return CamOfName(aName);
}

ElRotation3D cNewO_NameManager::OriCam2On1(const std::string & aNOri1,const std::string & aNOri2,bool & OK) const
{
    OK = false;
    bool aN1InfN2 = (aNOri1<aNOri2);
    std::string aN1 =  aN1InfN2?aNOri1:aNOri2;
    std::string aN2 =  aN1InfN2?aNOri2:aNOri1;

    if (!  ELISE_fp::exist_file(NameXmlOri2Im(aN1,aN2,true)))
       return ElRotation3D::Id;


    cXml_Ori2Im  aXmlO = GetOri2Im(aN1,aN2);
    OK = aXmlO.Geom().IsInit();
    if (!OK)
       return ElRotation3D::Id;
    const cXml_O2IRotation & aXO = aXmlO.Geom().Val().OrientAff();
    ElRotation3D aR12 =    ElRotation3D (aXO.Centre(),ImportMat(aXO.Ori()),true);

    OK = true;
    return aN1InfN2 ? aR12.inv() : aR12;
    //  return aN1InfN2 ? aR12 : aR12.inv();
}

std::pair<ElRotation3D,ElRotation3D> cNewO_NameManager::OriRelTripletFromExisting
                                     (
                                                    const std::string & aInOri,
                                                    const std::string & aN1,
                                                    const std::string & aN2,
                                                    const std::string & aN3,
                                                    bool & Ok
                                      )
{
    std::pair<ElRotation3D,ElRotation3D> aRes(ElRotation3D::Id,ElRotation3D::Id);
    Ok = false;
    CamStenope * aCam1 = CamOriOfNameSVP(aN1,aInOri);
    CamStenope * aCam2 = CamOriOfNameSVP(aN2,aInOri);
    CamStenope * aCam3 = CamOriOfNameSVP(aN3,aInOri);
    if (aCam1 && aCam2 && aCam3)
    {
       ElRotation3D aRot2Sur1  = (aCam2->Orient() *aCam1->Orient().inv());
       ElRotation3D aRot3Sur1  = (aCam3->Orient() *aCam1->Orient().inv());
       aRot2Sur1 = aRot2Sur1.inv();
       aRot3Sur1 = aRot3Sur1.inv();

       double aDist = euclid(aRot2Sur1.tr());
       aRot2Sur1.tr() = aRot2Sur1.tr() /aDist;
       aRot3Sur1.tr() = aRot3Sur1.tr() /aDist;
       Ok = true;
       aRes.first = aRot2Sur1;
       aRes.second = aRot3Sur1;
    }
    return aRes;
}



cResVINM::cResVINM() :
   mCam1    (0),
   mCam2    (0),
   mHom     (cElHomographie::Id()),
   mResHom  (-1)
{
}

cResVINM cNewO_NameManager::ResVINM(const std::string & aN1,const std::string & aN2) const
{
    cResVINM  aRes;
    bool Ok;
    ElRotation3D aR2On1 = OriCam2On1(aN1,aN2,Ok);
    if (!Ok)
    {
        return aRes;
    }

    CamStenope* aCam1 = CalibrationCamera(aN1)->Dupl();
    CamStenope* aCam2 = CalibrationCamera(aN2)->Dupl();

    aCam1->SetOrientation(ElRotation3D::Id);
    aCam2->SetOrientation(aR2On1);

    aRes.mCam1 = aCam1;
    aRes.mCam2 = aCam2;

    cXml_Ori2Im  aXmlO = GetOri2Im(aN1,aN2);
    aRes.mHom = cElHomographie(aXmlO.Geom().Val().HomWithR().Hom());
    aRes.mResHom = aXmlO.Geom().Val().HomWithR().ResiduHom();

    return aRes;
}


std::pair<CamStenope*,CamStenope*> cNewO_NameManager::CamOriRel(const std::string & aN1,const std::string & aN2) const
{
    cResVINM aRV =   ResVINM(aN1,aN2);

    return  std::pair<CamStenope*,CamStenope*>(aRV.mCam1,aRV.mCam2);

}



//   cXml_Ori2Im GetOri2Im(const std::string & aN1,const std::string & aN2);

std::string cNewO_NameManager::NameOriOut(const std::string & aNameIm) const
{
   return mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+OriOut(),aNameIm,true);
}

CamStenope *  cNewO_NameManager::OutPutCamera(const std::string & aName) const
{
    return  mICNM->StdCamStenOfNames(aName,OriOut());
}


/*
*/
CamStenope *  cInterfChantierNameManipulateur::StdCamStenOfNames(const std::string & aNameIm,const std::string & anOri)
{

     std::string aKey = "NKS-Assoc-Im2Orient@-"+ anOri ;
     std::string aNameCam =  Assoc1To1(aKey,aNameIm,true);
     return CamOrientGenFromFile(aNameCam,this);
}

CamStenope *  cInterfChantierNameManipulateur::StdCamStenOfNamesSVP(const std::string & aNameIm,const std::string & anOri)
{

     std::string aKey = "NKS-Assoc-Im2Orient@-"+ anOri ;
     std::string aNameCam =  Assoc1To1(aKey,aNameIm,true);
     if (! ELISE_fp::exist_file(aNameCam))
        return 0;
     return CamOrientGenFromFile(aNameCam,this);
}

CamStenope * cNewO_NameManager::CamOriOfName(const std::string & aNameIm,const std::string & anOri)
{
    return mICNM->StdCamStenOfNames(aNameIm,anOri);
}

CamStenope * cNewO_NameManager::CamOriOfNameSVP(const std::string & aNameIm,const std::string & anOri)
{
    std::string aKey = "NKS-Assoc-Im2Orient@-"+ anOri ;
    std::string aNameCam =  mICNM->Assoc1To1(aKey,aNameIm,true);
    if (! ELISE_fp::exist_file(aNameCam))
       return 0;
    return CamOrientGenFromFile(aNameCam,mICNM);
}


const std::string &   cNewO_NameManager::OriCal() const {return mPrefOriCal;}



std::string cNewO_NameManager::NameXmlOri2Im(cNewO_OneIm* aI1,cNewO_OneIm* aI2,bool Bin) const
{
    return NameXmlOri2Im(aI1->Name(),aI2->Name(),Bin);
}

std::string cNewO_NameManager::NameXmlOri2Im(const std::string & aN1,const std::string & aN2,bool Bin) const
{
    return Dir3POneImage(aN1,true) + "OriRel-" + aN2 +  (Bin ? ".dmp" : ".xml");
}
/*
*/

std::string cNewO_NameManager::NameListeImOrientedWith(const std::string & aName,bool Bin) const
{
    return Dir3POneImage(aName) + "ListOrientedsWith-" + aName + (Bin ? ".dmp" : ".xml");
}


std::string cNewO_NameManager::RecNameListeImOrientedWith(const std::string & aName,bool Bin) const
{
    return Dir3POneImage(aName) + "RecListOrientedsWith-" + aName + (Bin ? ".dmp" : ".xml");
}



std::string cNewO_NameManager::NameListeCpleOriented(bool Bin) const
{
    return Dir3P() + "ListCpleOriented"+ (Bin ? ".dmp" : ".xml");
}

std::string cNewO_NameManager::NameListeCpleConnected(bool Bin) const
{
    return Dir3P() + "ListCpleConnected"+ (Bin ? ".dmp" : ".xml");
}


std::string cNewO_NameManager::NameRatafiaSom(const std::string & aName,bool Bin) const
{
   return  Dir3POneImage(aName) + "Ratafia." + ExtXml(Bin);
}



std::list<std::string>  cNewO_NameManager::ListeImOrientedWith(const std::string & aName) const
{
   return StdGetFromPCP(NameListeImOrientedWith(aName,true),ListOfName).Name();
}

std::list<std::string>  cNewO_NameManager::Liste2SensImOrientedWith(const std::string & aName) const
{
    std::list<std::string>  aRes = ListeImOrientedWith(aName);
    std::list<std::string> aResRec =  StdGetFromPCP(RecNameListeImOrientedWith(aName,true),ListOfName).Name();

    std::copy(aResRec.begin(),aResRec.end(),std::back_inserter(aRes));
    return aRes;
}


std::list<std::string>  cNewO_NameManager::ListeCompleteTripletTousOri(const std::string & aN1,const std::string & aN2) const
{
    cListOfName aL1 = StdGetFromPCP(NameListeImOrientedWith(aN1,true),ListOfName);
    cListOfName aL2 = StdGetFromPCP(NameListeImOrientedWith(aN2,true),ListOfName);

    std::set<std::string>  aS1(aL1.Name().begin(),aL1.Name().end());


    std::list<std::string> aRes;

    for  (std::list<std::string>::const_iterator it2=aL2.Name().begin() ; it2!=aL2.Name().end() ; it2++)
        if (DicBoolFind(aS1,*it2))
           aRes.push_back(*it2);

    return  aRes;
}




/*
*/

std::string cNewO_NameManager::NameTimingOri2Im() const
{
    return  Dir3P(true)  + "Timing2Im.xml";
}

cXml_Ori2Im cNewO_NameManager::GetOri2Im(const std::string & aN1,const std::string & aN2) const
{
   return StdGetFromSI(mDir + NameXmlOri2Im(aN1,aN2,true),Xml_Ori2Im);
}


/************************  TRIPLETS *****************/

std::string  cNewO_NameManager::Dir3P(bool WithMakeDir) const
{
    std::string aRes = mDir + mDirTmp ;
    if (WithMakeDir)  ELISE_fp::MkDir(aRes);
    return aRes;
}

std::string  cNewO_NameManager::Dir3POneImage(const std::string & aName,bool WithMakeDir) const
{
    std::string aRes = Dir3P(WithMakeDir) + aName + "/";
    if (WithMakeDir)  ELISE_fp::MkDir(aRes);
    return aRes;
}

std::string  cNewO_NameManager::Dir3POneImage(cNewO_OneIm * anIm,bool WithMakeDir) const
{
    return Dir3POneImage(anIm->Name(),WithMakeDir);
}




std::string  cNewO_NameManager::Dir3PDeuxImage(const std::string & aName1,const std::string & aName2,bool WithMakeDir)
{
    std::string aRes = Dir3POneImage(aName1,WithMakeDir) + aName2 + "/";
    if (WithMakeDir)  ELISE_fp::MkDir(aRes);
    return aRes;
}


std::string  cNewO_NameManager::Dir3PDeuxImage(cNewO_OneIm * anI1,cNewO_OneIm * anI2,bool WithMakeDir)
{
   return Dir3PDeuxImage(anI1->Name(),anI2->Name(),WithMakeDir);
}




std::string cNewO_NameManager::NameHomFloat(const std::string & aName1,const std::string & aName2)
{
   return Dir3PDeuxImage(aName1,aName2,false) + "HomFloatSym"  + ".dat";
}

std::string cNewO_NameManager::NameHomFloat(cNewO_OneIm * anI1,cNewO_OneIm * anI2)
{
    return NameHomFloat(anI1->Name(),anI2->Name());
}





std::string cNewO_NameManager::NameTripletsOfCple(cNewO_OneIm * anI1,cNewO_OneIm * anI2,bool ModeBin)
{
     return  Dir3PDeuxImage(anI1,anI2,false) + "ImsOfTriplets." + std::string(ModeBin ? "dmp" : "xml");
}


//========================

std::string cNewO_NameManager::NameAttribTriplet
            (
               const std::string & aPrefix,const std::string & aPost,
               const std::string &   aI1,const std::string &   aI2,const std::string &   aI3,
               bool WithMakeDir
            )

{
    ELISE_ASSERT(aI1<aI2,"cNO_P3_NameM::NameAttribTriplet");
    ELISE_ASSERT(aI2<aI3,"cNO_P3_NameM::NameAttribTriplet");

    std::string aDir = Dir3PDeuxImage(aI1,aI2,WithMakeDir);

    return aDir + "Triplet-" + aPrefix + "-" + aI3 + "." + aPost;
}
/*
*/

std::string cNewO_NameManager::NameAttribTriplet
            (
               const std::string & aPrefix,const std::string & aPost,
               cNewO_OneIm * aI1,cNewO_OneIm * aI2,cNewO_OneIm * aI3,
               bool WithMakeDir
            )

{
    return NameAttribTriplet(aPrefix,aPost,aI1->Name(),aI2->Name(),aI3->Name(),WithMakeDir);
}


std::string cNewO_NameManager::NameHomTriplet(const std::string & aI1,const std::string & aI2,const std::string & aI3,bool WithMakeDir)
{
    return NameAttribTriplet("Hom","dat",aI1,aI2,aI3,WithMakeDir);
}

std::string cNewO_NameManager::NameHomTriplet(cNewO_OneIm *aI1,cNewO_OneIm *aI2,cNewO_OneIm *aI3,bool WithMakeDir)
{
    return NameAttribTriplet("Hom","dat",aI1->Name(),aI2->Name(),aI3->Name(),WithMakeDir);
}






std::string cNewO_NameManager::NameOriInitTriplet(bool ModeBin,cNewO_OneIm *aI1,cNewO_OneIm *aI2,cNewO_OneIm *aI3,bool WithMakeDir)
{
    return NameAttribTriplet("Ori0",(ModeBin ? "dmp" : "xml"),aI1,aI2,aI3,WithMakeDir);
}

std::string cNewO_NameManager::NameOriOptimTriplet(bool ModeBin,cNewO_OneIm *aI1,cNewO_OneIm *aI2,cNewO_OneIm *aI3,bool WithMakeDir)
{
    return NameAttribTriplet("OriOpt",(ModeBin ? "dmp" : "xml"),aI1,aI2,aI3,WithMakeDir);
}

std::string cNewO_NameManager::NameOriOptimTriplet(bool ModeBin,const std::string & aN1,const std::string & aN2,const std::string & aN3,bool WithMakeDir)
{
    return NameAttribTriplet("OriOpt",(ModeBin ? "dmp" : "xml"),aN1,aN2,aN3,WithMakeDir);
}



std::string cNewO_NameManager::NameTopoTriplet(bool aModeBin)
{
    return Dir3P() + "ListeTriplets" + mPrefOriCal +"." + (aModeBin ? "dmp" : "xml");
}

std::string cNewO_NameManager::NameCpleOfTopoTriplet(bool aModeBin)
{
    return Dir3P() + "ListeCpleOfTriplets" + mPrefOriCal +"." + (aModeBin ? "dmp" : "xml");
}



std::string cNewO_NameManager::NameOriGenTriplet(bool Quick,bool ModeBin,cNewO_OneIm *aI1,cNewO_OneIm *aI2,cNewO_OneIm *aI3)
{
   return Quick                                    ?
          NameOriInitTriplet (ModeBin,aI1,aI2,aI3) :
          NameOriOptimTriplet(ModeBin,aI1,aI2,aI3) ;
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
