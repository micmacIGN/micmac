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
#include "ReducHom.h"



cXmlRHHResLnk  ComputeHomographie
               (
                  const std::string & aName,
                  const std::string & aNameCal1,
                  const std::string & aNameCal2,
                  const std::string & aNameVerif1,
                  const std::string & aNameVerif2
               )
{
  cXmlRHHResLnk  aRes;

  ElPackHomologue aPack = ElPackHomologue::FromFile(aName);


  // Si les orientation de verification sont passees, on met des homologues correspondant a des 
  // homographies pures. On le fait en mettant tout le monde en Z=0, suppose une bascule faite
  if (aNameVerif1 != "")
  {
      ElPackHomologue aNewPack;
      CamStenope *  aCam1 = CamOrientGenFromFile(aNameVerif1,0);
      CamStenope *  aCam2 = CamOrientGenFromFile(aNameVerif2,0);
      for 
      (
          ElPackHomologue::iterator itH=aPack.begin();
          itH !=  aPack.end();
          itH++
      )
      {
          Pt3dr  aPTer = aCam1->PseudoInter(itH->P1(),*aCam2,itH->P2());
          aPTer.z = 0;
          Pt2dr aP1 = aCam1->R3toF2(aPTer);
          Pt2dr aP2 = aCam2->R3toF2(aPTer);
          if ( aCam1->IsInZoneUtile(aP1)  && aCam2->IsInZoneUtile(aP2))
          {
              aNewPack.Cple_Add(ElCplePtsHomologues(aP1,aP2));
          }

      }

      // std::cout << "SZZZ-VERIF " << aPack.size() << " => " << aNewPack.size() << "\n";

      aPack = aNewPack;
  }




  CamStenope *  aCam1 = CamOrientGenFromFile(aNameCal1,0);
  CamStenope *  aCam2 = CamOrientGenFromFile(aNameCal2,0);

  // On laisse, au cas ou l'on reveuille tester ....
  // On fait une correction de distorsion
  bool 	aCorCam = true;
  ELISE_ASSERT(aCorCam,"Cor cam mandatory");
  std::vector<Pt2dr> aVP1;
 

  if (aCorCam)
  {
      for 
      (
          ElPackHomologue::iterator itH=aPack.begin();
          itH !=  aPack.end();
          itH++
      )
      {
          itH->P1() =  aCam1->F2toPtDirRayonL3(itH->P1());
          itH->P2() =  aCam2->F2toPtDirRayonL3(itH->P2());
          aVP1.push_back(itH->P1());
      }
  }


  double anEcart;
  cElHomographie   aHom = cElHomographie::RobustInit(anEcart,&(aRes.Qual()),aPack,aRes.Ok(),NB_RANSAC_H,90.0,1000);
  int aNbPts = aPack.size();
  aRes.NbPts() = aNbPts;

  if (0)
  {
    cXmlHomogr aXH =  aHom.ToXml();

    std::cout << "ROBUST:Nan " << aHom.HasNan() << " " << aName << "\n";
    aHom.Show();
    std::cout << "XML ZX  " << aXH.Z().CoeffX() << "\n";
    MakeFileXML(aXH,"toto.xml");
    getchar();
  }

  if (aCorCam)
  {
      aRes.Qual() *=  aCam2->Focale();
  }




  if (aRes.Ok())
  {
     aRes.Hom12() = aHom.ToXml();


     Pt2di aSz (3,3);
     if (aNbPts > 100)
        aSz = Pt2di(4,4);
     if (aNbPts > 500)
        aSz = Pt2di(5,5);
     if (aNbPts > 2000)
        aSz = Pt2di(6,6);

   // mEchantP1 = GetDistribRepresentative(mCdg1,aVP2,aSz);
     aRes.EchRepP1() = GetDistribRepreBySort(aVP1,aSz,aRes.PRep());
  }



  return aRes;
}


int RHHComputHom_main(int argc,char ** argv)
{

   std::string aNameHom,aNameRes;
   std::string aNameCal1,aNameCal2;
   std::string toto;
   

    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameHom,"Hom points")
                     << EAMC(aNameCal1,"Calib  1")
                     << EAMC(aNameCal2,"Calib  2")
                     << EAMC(aNameRes,"Name res "),
         LArgMain()  << EAM(toto,"toto",true)
   );


   cXmlRHHResLnk aXmlLnk = ComputeHomographie(aNameHom,aNameCal1,aNameCal2,"","");

   MakeFileXML(aXmlLnk,aNameRes);

   return 1;
}



NS_RHH_BEGIN


/*************************************************/
/*                                               */
/*                cLink2Img                      */
/*                                               */
/*************************************************/

cLink2Img::cLink2Img(cImagH * aSrce,cImagH * aDest,const std::string & aNameH) :
   mNbPts      (0),
   mNbPtsAttr  (0),
   mSrce       (aSrce),
   mDest       (aDest),
   mAppli      (mSrce->Appli()),
   mNameH      (aNameH),
   mQualHom    (0),
   mHom12      (cElHomographie::Id()),
   mPckLoaded  (false),
   mHomLoaded  (false),
   mOkHom      (false)
{
}


//  Gestion des noms 

std::string cLink2Img::NameHomol() const
{
    return mSrce->Appli().Dir()+mNameH;
}

std::string cLink2Img::NameXmlHomogr() const
{
    return mAppli.NameFileHomogr(*this);
}

std::string  cLink2Img::NameComHomogr() const
{
   std::string aNameRes = NameXmlHomogr();
   if (mAppli.SkipHomDone() && ELISE_fp::exist_file(aNameRes))
      return "";

   return      MM3dBinFile("TestLib") 
            +  std::string(" RHHComputHom ")   //  RHHComputHom_main
            +  NameHomol() 
            +  std::string(" ")
            +  mSrce->NameCalib()
            +  std::string(" ")
            +  mDest->NameCalib()
            +  std::string(" ")
            + aNameRes;
}


void cLink2Img::LoadXmlHom(const cXmlRHHResLnk & aXml)
{
     mHomLoaded = true;
     mOkHom = aXml.Ok();
     if (mOkHom)
     {
         mQualHom = aXml.Qual();
         mHom12 =  cElHomographie(aXml.Hom12());
         mNbPts     = aXml.NbPts();
         mEchantP1  = aXml.EchRepP1();
         mPRep1     = aXml.PRep();
     }
}


void cLink2Img:: LoadComHomogr()
{
    cXmlRHHResLnk aXmlLnk = StdGetFromSI(NameXmlHomogr(),XmlRHHResLnk);
    LoadXmlHom(aXmlLnk);
}

void  cLink2Img::LoadStatPts(bool ExigOk)
{
     // LoadPack();
     if (!mHomLoaded) 
     {
        cXmlRHHResLnk 	aXml = ComputeHomographie
                               (
                                    NameHomol(),
                                    mSrce->NameCalib(),
                                    mDest->NameCalib(),
                                    mSrce->NameVerif(),
                                    mDest->NameVerif()
                               );
        LoadXmlHom(aXml);
     }
       
     if (ExigOk)
     {
         ELISE_ASSERT(mOkHom,"Cannot get Hom12 : mOkHom=false");
     }
}

void  cLink2Img::LoadStatPts(bool ExigOk) const
{
   const_cast<cLink2Img*>(this)->LoadStatPts(ExigOk);
}

void cLink2Img::LoadPtsHom()
{
   LoadStatPts(true);
   if (mPckLoaded)
      return;

   ELISE_ASSERT(false,"Currenlty RHH shoul not need to load pack");
   mPckLoaded = true;
   mPack = ElPackHomologue::FromFile(mSrce->Appli().Dir()+mNameH);

   CamStenope * aCam1 = mSrce->CamC();
   CamStenope * aCam2 = mDest->CamC();
   for
   (
       ElPackHomologue::iterator itP=mPack.begin();
       itP != mPack.end();
       itP++
   )
   {
       itP->P1() =  aCam1->F2toPtDirRayonL3(itP->P1());
       itP->P2() =  aCam2->F2toPtDirRayonL3(itP->P2());
       // mEchantP1.push_back(Pt3dr(aP1.x,aP1.y,1.0));
   }
}

void  cLink2Img::LoadPtsHom() const
{
   const_cast<cLink2Img*>(this)->LoadPtsHom();
}


const ElPackHomologue & cLink2Img::Pack() const
{
   LoadPtsHom();
   return mPack;
}
ElPackHomologue & cLink2Img::Pack() 
{
   LoadPtsHom();
   return mPack;
}


// Compute the quality of the "closing" relation
//     mDest->Hi2t() * mHom12 = mSrce->Hi2t()
//     This quality is estimated on the weighted set mEchantP1


double cLink2Img::CoherenceH()
{
    LoadStatPts(true);

    cElHomographie aI2T_A = CalcSrceFromDest();
    cElHomographie aI2T_B = mSrce->Hi2t();

    double aSomP=0;
    double aSomDist=0;

    for (int aKP=0 ; aKP < int(mEchantP1.size()) ; aKP++)
    {
         Pt3dr aP3(mEchantP1[aKP]);
         double aPds (aP3.z);
         Pt2dr aP1 (aP3.x,aP3.y);

         Pt2dr aPA = aI2T_A.Direct(aP1);
         Pt2dr aPB = aI2T_B.Direct(aP1);
         double aDist = square_euclid(aPA-aPB);

         aSomP += aPds;
         aSomDist  += aDist * aPds;
    }

    return sqrt(aSomDist/aSomP);
}



cImagH * cLink2Img::Srce() const
{
   return mSrce;
}
cImagH * cLink2Img::Dest() const
{
   return mDest;
}

const std::string &  cLink2Img::NameH() const
{
    return mNameH;
}


const int   &  cLink2Img::NbPts() const
{
    LoadStatPts(true);
    return mNbPts;
}

/*
int   &  cLink2Img::NbPtsAttr()
{
    return mNbPtsAttr;
}
*/

const bool & cLink2Img::OkHom() const
{
   LoadStatPts(false);
   return mOkHom;
}


const double & cLink2Img::QualHom() const
{
   LoadStatPts(true);
   return mQualHom;
}

const cElHomographie &  cLink2Img::Hom12() const
{
    LoadStatPts(true);
    return mHom12;
}
cElHomographie &  cLink2Img::Hom12() 
{
    LoadStatPts(true);
    return mHom12;
}




cElHomographie cLink2Img::CalcSrceFromDest ()
{
    // return mHom12 * mSrce->Ht2i(); A CHANGER
    return mDest->Hi2t() * Hom12();
}

const std::vector<Pt3dr> & cLink2Img::EchantP1() const
{
    LoadStatPts(true);
    return mEchantP1;
}


cEqHomogFormelle * &  cLink2Img::EqHF()
{
    return mEqHF;
}


double cLink2Img::PdsEchant() const
{
   double  aSomP = 0;
   for (int aKP=0 ; aKP<int(mEchantP1.size()) ; aKP++)
       aSomP += mEchantP1[aKP].z;
   return aSomP;
}




/*************************************************/
/*                                               */
/*                 cImagH                        */
/*                                               */
/*************************************************/


cImagH::cImagH(const std::string & aName,cAppliReduc & anAppli,int aNum) :
   mAppli    (anAppli),
   mName     (aName),
   mNameCalib (mAppli.NameCalib(mName)),
   mCamC      (CamOrientGenFromFile(mNameCalib,0)),
   mNameVerif (mAppli.NameVerif(mName)),
   mNum      (aNum),
   mNumTmp   (-1),
   mSomQual  (0),
   mSomNbPts (0),
   mH2ImC     (cElHomographie::Id()),
   mHi2t     (cElHomographie::Id()),
   mHTmp     (cElHomographie::Id()),
   mMDP      (cMetaDataPhoto::CreateExiv2(mAppli.Dir() + mName)),
   mLnkClosed  (false),
   mEqOneHF    (0),
   mC2CI       (false)
{

   // std::cout << "NCCC " << mNameCalib << " [" << mNameVerif<< "]\n";
}

void cImagH::AssertLnkClosed() const
{
  ELISE_ASSERT(mLnkClosed,"cLink2Img::AssertClosed");
}
void cImagH::AssertLnkUnclosed() const
{
  ELISE_ASSERT(!mLnkClosed,"cLink2Img::AssertUnclosed");
}


CamStenope *   cImagH::CamC()
{
   return mCamC;
}

cEqOneHomogFormelle * &  cImagH::EqOneHF()
{
   return mEqOneHF;
}

bool  &  cImagH::C2CI()
{
   return mC2CI;
}

   //============ FONCTION DE GRAPHE IMAGE =========================

cLink2Img * cImagH::GetLinkOfImage(cImagH* anI2)
{
   AssertLnkClosed();
   tMapName2Link::iterator anIt = mLnks.find(anI2);
   if (anIt==mLnks.end())
      return 0;
   return anIt->second;
}


void cImagH::AddLink(cImagH * anI2,const std::string & aNameH)
{
      AssertLnkUnclosed();
      mLnks[anI2] = new cLink2Img(this,anI2,aNameH);
}

void cImagH::SetMarqued(int aK)
{
   mMarques.set_kth_true(aK);
}
void cImagH::SetUnMarqued(int aK)
{
   mMarques.set_kth_false(aK);
}
bool cImagH::Marqued(int aK) const
{
   return mMarques.kth(aK);
}

cElHomographie &   cImagH::Hi2t()  { return mHi2t; }
cElHomographie &   cImagH::HTmp()  { return mHTmp; }
cElHomographie &   cImagH::H2ImC() { return mH2ImC; }



   //============ FUSION DE POINT =========================

const std::vector<cLink2Img*> &  cImagH::VLink() const
{
   AssertLnkClosed();

   return mVLnkInterneSorted;
}


const std::string & cImagH::Name() const
{
  return mName;
}

const std::string & cImagH::NameCalib() const
{
  return mNameCalib;
}

const std::string & cImagH::NameVerif() const
{
  return mNameVerif;
}

void cImagH::AddOnePtToExistingH(cPtHom * aH1,const Pt2dr & aP1,cImagH * aI2,const Pt2dr & aP2)
{
    aH1->AddMesureInImage(aI2,aP2);
    // aI2->mMapH[aP2] = aH1;
}


void  cImagH::FusionneIn(cPtHom *aH1,const Pt2dr & aP1,cImagH *aI2,cPtHom *aH2,const Pt2dr & aP2)
{

   aH1->OkAbsorb(aH2);
   // aI2->mMapH[aP2] = aH1;
   // if
}

void  cImagH::SetPHom(const Pt2dr & aP,cPtHom * aH)
{
   mMapH[aP] = aH;
}

void cImagH::AddOnePair(const Pt2dr & aP1,cImagH * aI2,const Pt2dr & aP2)
{
    std::map<Pt2dr,cPtHom *>::iterator it1 = mMapH.find(aP1);
    std::map<Pt2dr,cPtHom *>::iterator it2 = aI2->mMapH.find(aP2);

    if ((it1==  mMapH.end()) && (it2==  aI2->mMapH.end()))
    {
        cPtHom * aH = cPtHom::NewGerm(this,aP1,aI2,aP2);
        mMapH[aP1] = aH;
        aI2->mMapH[aP2] = aH;
    }
    else if ((it1!= mMapH.end()) && (it2==  aI2->mMapH.end()))
    {
       it1->second->AddMesureInImage(aI2,aP2);
       // AddOnePtToExistingH(it1->second,aP1,aI2,aP2);
    }
    else if ((it1 == mMapH.end()) && (it2!=aI2->mMapH.end()))
    {
       it2->second->AddMesureInImage(this,aP1);
       // aI2->AddOnePtToExistingH(it2->second,aP2,this,aP1);
    }
    else if (it1->second==it2->second)
    {
         it1->second->IncrCptArc();
    }
    else
    {
         if (it1->second->NbIm() >= it2->second->NbIm())
            FusionneIn(it1->second,aP1,aI2,it2->second,aP2);
         else
            aI2->FusionneIn(it2->second,aP2,this,it1->second,aP1);
    }
}



void  cImagH::ComputePtsLink(cLink2Img & aLnk)
{
    const ElPackHomologue & aPack = aLnk.Pack();

    // std::cout << "    LNK " << aLnk.NameH() << "  " << aPack.size() << "\n";
    for
    (
       ElPackHomologue::const_iterator itP=aPack.begin();
       itP != aPack.end();
       itP++
    )
    {
        //  std::cout << itP->P1() << " " << itP->P2() << "\n";
         AddOnePair(itP->P1(),aLnk.Dest(),itP->P2());
    }
}


void cImagH::ComputePts()
{
    std::cout << "Compute " << mName << "\n";


    //  D'abod on complete avec les arcs
    for ( tMapName2Link::iterator itL=mLnks.begin(); itL!=mLnks.end(); itL++)
    {
        ComputePtsLink(*(itL->second));
    }
}

std::string cImagH::NameOriHomPlane() const
{
   return mAppli.Dir() + "RHH/" + mName +  "/OriPlane-" +  mName  + ".xml";
}


cHomogFormelle *  & cImagH::HF()
{
   return mHF;
}

cAppliReduc &     cImagH::Appli()
{
   return mAppli;
}

const tMapName2Link & cImagH::Lnks() const { return mLnks; }

double & cImagH::GainLoc() {return mGainLoc;}
bool & cImagH::InitLoc() {return mInitLoc;}


cLink2Img *  cImagH::GetLnkKbrd(int & aK)
{
    std::cout << "For " << mName << " ENTER NUMBER FOR LINK between 0 and " << VLink().size() << " [" << VLink().front()->Dest()->Name()  << " / "<<  VLink().back()->Dest()->Name()<<"]\n";

    cin >> aK;
    ELISE_ASSERT((aK>=0) && (aK<int(VLink().size())),"cImagH::GetLnkKbrd");
    cLink2Img *  aRes = VLink()[aK];
    std::cout <<  "Got " << aRes->Dest()->Name() << "\n";

    return aRes;
}



NS_RHH_END

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
