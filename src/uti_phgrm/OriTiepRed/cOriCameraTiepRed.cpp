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

#include "OriTiepRed.h"

NS_OriTiePRed_BEGIN


/**********************************************************************/
/*                                                                    */
/*                         cCameraTiepRed                             */
/*                                                                    */
/**********************************************************************/

Pt2dr cCameraTiepRed::ToImagePds(const Pt2dr & aP) const
{
   return aP / mResolPds;
}

cCameraTiepRed::cCameraTiepRed
(
    cAppliTiepRed &         anAppli,
    const std::string &     aName,
    CamStenope *            aCamOr,
    CamStenope *            aCamCal,
    bool                    IsMaster
) :
   mAppli        (anAppli),
   mNameIm       (aName),
   mMTD          (cMetaDataPhoto::CreateExiv2(anAppli.Dir() + aName)),
   mCsOr         (aCamOr),
   mCsCal        (aCamCal),
   mNbPtsHom2Im  (0),
   mNum          (-1),
   mXRat         (0),
   mIsMaster     (IsMaster),
   mMasqIsDone   (false),
   mSzIm         (mMTD.TifSzIm()),
   mResolPds     (anAppli.ModeIm() ? 20.0 : 200.0),
   mSzPds        (round_up(Pt2dr(mSzIm)/mResolPds)),
   mIMasqM       (mSzPds.x,mSzPds.y,0),
   mTMasqM       (mIMasqM),
   mAlive        (true)
{
   if (mAppli.FromRatafiaBox())
   {
        mXRat = new cXml_RatafiaSom(StdGetFromSI(mAppli.NM().NameRatafiaSom(aName,true),Xml_RatafiaSom));
   }
}

cAppliTiepRed & cCameraTiepRed::Appli() {return mAppli;}

bool cCameraTiepRed::Alive() const 
{
   return mAlive;
}

void cCameraTiepRed::SetDead()
{
   mAlive = false;
}


CamStenope  & cCameraTiepRed::CsOr() 
{
   ELISE_ASSERT(mCsOr!=0,"cCameraTiepRed::CsOr");
   return *mCsOr;
}
const int & cCameraTiepRed::NbPtsHom2Im() const {return mNbPtsHom2Im;}
bool  cCameraTiepRed::SelectOnHom2Im() const
{
    return mNbPtsHom2Im >= mAppli.ThresholdTotalNbPts2Im();
}

void cCameraTiepRed::SetNum(int aNum)
{
   mNum = aNum;
}
const int & cCameraTiepRed::Num() const
{
   return mNum;
}


Pt2dr cCameraTiepRed::Hom2Cam(const Pt2df & aP) const
{
     Pt3dr aQ(aP.x,aP.y,1.0);
     return mCsCal->L3toF2(aQ);
}

Pt2dr cCameraTiepRed::Hom2Pds(const Pt2df & aP) const
{
    return ToImagePds(Hom2Cam(aP));
}



const std::string cCameraTiepRed::NameIm() const { return mNameIm; }

Pt3dr cCameraTiepRed::BundleIntersection(const Pt2df & aPH1,const cCameraTiepRed & aCam2,const Pt2df & aPH2,double & Precision) const
{

     Pt2dr aPC1 = Hom2Cam(aPH1);
     Pt2dr aPC2 = aCam2.Hom2Cam(aPH2);

     ElSeg3D aSeg1 = mCsOr->Capteur2RayTer(aPC1);
     ElSeg3D aSeg2 = aCam2.mCsOr->Capteur2RayTer(aPC2);

     bool Ok;
     double aD;
     Pt3dr aRes= InterSeg(aSeg1.P0(),aSeg1.P1(),aSeg2.P0(),aSeg2.P1(),Ok,&aD);

     if (Ok)
     {
         Pt2dr aRP1 = mCsOr->Ter2Capteur(aRes);
         Pt2dr aRP2 = aCam2.mCsOr->Ter2Capteur(aRes);
         Precision = (euclid(aRP1-aPC1)+euclid(aRP2-aPC2)) / 2.0;
     }
     else
     {
        Precision = 1e20;
     }
     
     return aRes;
    
}




void cCameraTiepRed::LoadHomCam(cCameraTiepRed & aCam2)
{

    ELISE_ASSERT(mNameIm < aCam2.mNameIm,"cCameraTiepRed::LoadHom order name");
    cCameraTiepRed * aMaster = 0;
    cCameraTiepRed * aSlave = 0;

    if (mIsMaster)
    {
         aMaster = this;
         aSlave = & aCam2;
    }
    else if (aCam2.mIsMaster)
    {
         aMaster = & aCam2;
         aSlave =  this;
    }


    Im2D_REAL4          aImPdsP(mSzPds.x,mSzPds.y,0.0);
    TIm2D<REAL4,REAL8>  aTImPdsP(aImPdsP);
    int                 aNbP=0;
    Im2D_REAL4          aImPdsM(mSzPds.x,mSzPds.y,0.0);
    TIm2D<REAL4,REAL8>  aTImPdsM(aImPdsM);
    int                 aNbM =0;

    // Declare Input Tie Points
    std::vector<Pt2df> aVPIn1,aVPIn2;
    // Load Input
    mAppli.NM().LoadHomFloats(NameIm(),aCam2.NameIm(),&aVPIn1,&aVPIn2);  // would have worked for I2 > I1 
    Box2dr aBox = mAppli.ParamBox().Box();
    double aThresh = mAppli.ThresoldPrec2Point();

     // Create a connexion with initialy no tie points
    cLnk2ImTiepRed * aLnk = new cLnk2ImTiepRed(this,&aCam2);
    std::vector<Pt2df> & aVPOut1 = aLnk->VP1();
    std::vector<Pt2df> & aVPOut2 = aLnk->VP2();

    // Filter the ties points that are inside the current tiles and
    // have "good" intersection

    // double aSomRes = 0;
    // double aNbRes = 0;
    std::vector<double> aVRes;
    for (int aKP=0 ; aKP<int(aVPIn1.size()) ; aKP++)
    {
        bool Ok= false;
        const Pt2df & aPf1 = aVPIn1[aKP];
        const Pt2df & aPf2 = aVPIn2[aKP];

        if (mAppli.ModeIm())
        {
             if (aMaster)
             {
                  Pt2df aPfM = (aMaster==this) ? aPf1 : aPf2;
                  Pt2df aPfS = (aMaster==this) ? aPf2 : aPf1;


                  Ok =  mAppli.ParamBox().BoxRab().inside(Pt2dr(aPfM.x,aPfM.y));
                  // Pt2dr  aP2S = aSlave->Hom2Cam(aPfS);
                  // Pt2dr aPPS = ToImagePds(aP2S);
                  Pt2dr aPPS = aSlave->Hom2Pds(aPfS);
                  if (Ok)
                  {
                     aNbP++;
                     aTImPdsP.incr(aPPS,1.0);
                  }
                  else
                  {
                     aNbM++;
                     aTImPdsM.incr(aPPS,1.0);
                  }
             }
             else
             {
                  double aSeuil = 65;
                  ELISE_ASSERT(aCam2.mMasqIsDone &&  mMasqIsDone ,"Incoh Masq Done");
                  Pt2dr aP1 = Hom2Pds(aPf1);
                  Pt2dr aP2 = Hom2Pds(aPf2);
                  Ok = (mTMasqM.getprojR(aP1) > aSeuil) || (aCam2.mTMasqM.getprojR(aP2) > aSeuil) ;
                  // Pt2dr  aP2S = aSlave->Hom2Cam(aPfS);
             }
        }
        else if (mAppli.OrLevel() >= eLevO_Glob)
        {
            double aD; // store the reprojection error
            Pt3dr aPTer = BundleIntersection(aPf1,aCam2,aPf2,aD);

            Ok = (aD< aThresh) && aBox.inside(Pt2dr(aPTer.x,aPTer.y));
        }
        else
        {
           Ok = true;
        }

        if ( Ok)
        {
            aVPOut1.push_back(aVPIn1[aKP]);
            aVPOut2.push_back(aVPIn2[aKP]);
        }
    }

    if (aMaster)
    {
         double 	DMoy = sqrt((mSzPds.x*mSzPds.y) / double(aNbP + aNbP));
         FilterGauss(aImPdsP,DMoy*2);
         FilterGauss(aImPdsM,DMoy*2);

         ELISE_COPY(mIMasqM.all_pts(),255*(aImPdsP.in()/ Max(aImPdsM.in()+aImPdsP.in(),1e-10)), aSlave->mIMasqM.out());
         // Tiff_Im::CreateFromIm(mIMasqM,"MasqRatafia-"+aCam2.NameIm() + ".tif");
         aSlave->mMasqIsDone = true;
    }

    if (mAppli.ModeIm())
    {
       std::string aNH = mAppli.NameHomolGlob(mNameIm,aCam2.mNameIm);

       if (ELISE_fp::exist_file(aNH) && mAppli.UsePrec())
       {
         
           mAppli.NM().GenLoadHomFloats(aNH,&(aLnk->VPPrec1()),&(aLnk->VPPrec2()),false);  // would have worked for I2 > I1 
       }
    }

    // If enough tie point , memorize the connexion 

    if (int(aVPOut1.size()) >= mAppli.ThresholdNbPts2Im())
    {
        // Update counters
        mNbPtsHom2Im +=  aVPOut1.size();  
        aCam2.mNbPtsHom2Im +=  aVPOut1.size();
        // Ask application to memorize
        mAppli.AddLnk(aLnk);
    }
    else
    {
         delete aLnk;
    }
}
  
void cCameraTiepRed::AddCamBox(cCameraTiepRed* aCam2,int aKBox)
{
   mMapCamBox[aCam2].push_back(aKBox);
}


void cCameraTiepRed::SaveHom(cCameraTiepRed* aCam2,const std::list<int> & aLBox)
{

    std::pair<CamStenope*,CamStenope*>  aPC((CamStenope *)NULL, (CamStenope *)NULL);
    if (mAppli.VerifNM())// (this != aCam2)
    {
       aPC = mAppli.NM().CamOriRel(NameIm(),aCam2->NameIm());
    }
    CamStenope* aCS1 = aPC.first;
    CamStenope* aCS2 = aPC.second;

    ElPackHomologue aRes;
    for (std::list<int>::const_iterator itI=aLBox.begin(); itI!=aLBox.end() ; itI++)
    {
         std::string aName = mAppli.NameHomol(NameIm(),aCam2->NameIm(),*itI);

         if (ELISE_fp::exist_file(aName))
         {
             ElPackHomologue aPack = ElPackHomologue::FromFile(aName);
             aRes.Add(aPack);

             // Verif
             if (aCS2)
             {
                 std::vector<double> aVD;
                 for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end(); itP++)
                 {
                     double aDist;
                     aCS1->PseudoInterPixPrec(itP->P1(),*aCS2,itP->P2(),aDist);
                     aVD.push_back(aDist);
                 }
                 if (aVD.size())
                     std::cout << "Verif   CamOriRel " << MedianeSup(aVD) << "\n";
             }

         }
    }

    if (aRes.size())
    {
         std::string aKeyH = "NKS-Assoc-CplIm2Hom@"+ mAppli.StrOut() + "@dat";
         std::string aNameH = mAppli.ICNM()->Assoc1To2(aKeyH,NameIm(),aCam2->NameIm(),true);
         aRes.StdPutInFile(aNameH);
         // std::string aNameH = mAppli
    }
}

void  cCameraTiepRed::SaveHom()
{
    for (std::map<cCameraTiepRed*,std::list<int> >::const_iterator itM=mMapCamBox.begin(); itM!=mMapCamBox.end() ;itM++)
    {
       SaveHom(itM->first,itM->second);
    }
}

bool  cCameraTiepRed::IsMaster() const
{
   return mIsMaster;
}

NS_OriTiePRed_END



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
