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

#include "TiepRed.h"


/**********************************************************************/
/*                                                                    */
/*                         cAppliTiepRed                              */
/*                                                                    */
/**********************************************************************/


cAppliTiepRed::cAppliTiepRed(int argc,char **argv)  :
     mPrec2Point              (5.0),
     mThresholdPrecMult       (2.0),
     mThresholdNbPts2Im       (3),
     mThresholdTotalNbPts2Im  (10),
     mSzTile                  (1600),
     mCallBack                (false)
{
   MMD_InitArgcArgv(argc,argv);

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatImage, "Name Image 1",  eSAM_IsPatFile),
         LArgMain()  << EAM(mCalib,"OriCalib",true,"Calibration folder if any")
                     << EAM(mPrec2Point,"Prec2P",true,"Threshold of precision for 2 Points")
                     << EAM(mKBox,"KBox",true,"Internal use")
                     << EAM(mSzTile,"SzTile",true,"Size of Tiles in Pixel")
   );
   mCallBack = EAMIsInit(&mKBox);
   mDir = DirOfFile(mPatImage);
   if (EAMIsInit(&mCalib))
   {
      StdCorrecNameOrient(mCalib,mDir);
   }
   const std::vector<std::string> * aFilesIm =0;
   if (mCallBack)
   {
       mXmlParBox = StdGetFromPCP(NameParamBox(mKBox),Xml_ParamBoxReducTieP);
       mBoxLoc = mXmlParBox.Box();
       aFilesIm = &(mXmlParBox.Ims());
   }
   else
   {
       cElemAppliSetFile anEASF(mPatImage);
       anEASF.Init(mPatImage);
       aFilesIm = anEASF.SetIm();
   }


   mSetFiles = new std::set<std::string>(aFilesIm->begin(),aFilesIm->end());
   std::cout << "## Get Nb Images " <<  aFilesIm->size() << "\n";


   mNM = cVirtInterf_NewO_NameManager::StdAlloc(mDir,mCalib);

   std::vector<double> aVResol;
   Pt2dr aPInf( 1E50, 1E50);
   Pt2dr aPSup(-1E50,-1E50);

   for (int aKI = 0 ; aKI<int(aFilesIm->size()) ; aKI++)
   {
       const std::string & aNameIm = (*aFilesIm)[aKI];
       CamStenope * aCS = mNM->OutPutCamera(aNameIm);
       cCameraTiepRed * aCam = new cCameraTiepRed(*this,aNameIm,aCS);
       
       mVecCam.push_back(aCam);
       mMapCam[aNameIm] = aCam;

       Box2dr aBox = aCam->CS().BoxSol();
       aPInf = Inf(aBox._p0,aPInf);
       aPSup = Sup(aBox._p1,aPSup);
       aVResol.push_back(aCam->CS().ResolutionSol());
       // std::cout << "BBB " << aBox._p0 << aBox._p1 << " " <<  aCam->CS().ResolutionSol() << "\n";
   }
   mBoxGlob = Box2dr(aPInf,aPSup);
   mResol = MedianeSup(aVResol);

   std::cout << "   BOX " << mBoxGlob << " Resol=" << mResol << "\n";
}


const std::string cAppliTiepRed::TheNameTmp = "Tmp-ReducTieP/";

std::string  cAppliTiepRed::NameParamBox(int aK) const
{
    return mDir+TheNameTmp + "Param_" +ToString(aK) + ".xml";
}

cVirtInterf_NewO_NameManager & cAppliTiepRed::NM(){ return *mNM ;}
const cXml_ParamBoxReducTieP & cAppliTiepRed::ParamBox() const {return mXmlParBox;}
const double & cAppliTiepRed::ThresoldPrec2Point() const {return  mPrec2Point;}
const int    & cAppliTiepRed::ThresholdNbPts2Im() const  {return mThresholdNbPts2Im;}
const int    & cAppliTiepRed::ThresholdTotalNbPts2Im() const  {return mThresholdTotalNbPts2Im;}
cCameraTiepRed * cAppliTiepRed::KthCam(int aK) {return mVecCam[aK];}
const double & cAppliTiepRed::ThresholdPrecMult() const {return mThresholdPrecMult;}


void cAppliTiepRed::AddLnk(cLnk2ImTiepRed * aLnk)
{
    mLnk2Im.push_back(aLnk);
}





void cAppliTiepRed::DoReduceBox()
{
   // Load Tie Points in box
   for (int aKI = 0 ; aKI<int(mVecCam.size()) ; aKI++)
   {
       cCameraTiepRed & aCam1 = *(mVecCam[aKI]);
       const std::string & anI1 = aCam1.NameIm();
       // Get list of images sharin tie-P with anI1
       std::list<std::string>  aLI2 = mNM->ListeImOrientedWith(anI1);
       for (std::list<std::string>::const_iterator itL= aLI2.begin(); itL!=aLI2.end() ; itL++)
       {
            const std::string & anI2 = *itL;
            // Test if the file anI2 is in the current pattern
            // As the martini result may containi much more file 
            if (mSetFiles->find(anI2) != mSetFiles->end())
            {
               // The result being symetric, the convention is that some data are stored only for  I1 < I2
               if (anI1 < anI2)
               {
                   cCameraTiepRed & aCam2 = *(mMapCam[anI2]);

                   aCam1.LoadHom(aCam2);
               }
            }
       }
   }

   // Select Cam ( and Link ) with enough points
   {
      std::vector<cCameraTiepRed *>  aNewV;
      int aNum=0;
      for (int aKC=0 ; aKC<int(mVecCam.size()) ; aKC++)
      {
          if (mVecCam[aKC]->SelectOnHom2Im())
          {
             aNewV.push_back(mVecCam[aKC]);
             mVecCam[aKC]->SetNum(aNum);
             aNum++;
          }
          else
          {
             mMapCam[mVecCam[aKC]->NameIm()] = 0;
             delete mVecCam[aKC];
          }
      }
      std::cout << "   CAMSS " << mVecCam.size() << " " << aNewV.size() << "\n";
      mVecCam = aNewV;

      std::list<cLnk2ImTiepRed *> aNewL;
      for (std::list<cLnk2ImTiepRed *>::const_iterator itL=mLnk2Im.begin() ; itL!=mLnk2Im.end() ; itL++)
      {
          if ((*itL)->Cam1().SelectOnHom2Im() && (*itL)->Cam2().SelectOnHom2Im())
             aNewL.push_back(*itL);
      }
      std::cout << "   LNK " << mLnk2Im.size() << " " << aNewL.size() << "\n";
      mLnk2Im = aNewL;
   }

   // merge topological tie point

    mMergeStruct  = new  tMergeStr(mVecCam.size());
    for (std::list<cLnk2ImTiepRed *>::const_iterator itL=mLnk2Im.begin() ; itL!=mLnk2Im.end() ; itL++)
    {
        (*itL)->Add2Merge(mMergeStruct);
    }
    mMergeStruct->DoExport();
    mLMerge =  & mMergeStruct->ListMerged();
    std::vector<int> aVHist(mVecCam.size(),0);

    double aSzTileAver = sqrt(mBoxLoc.surf()/mLMerge->size()); 

   // give ground coord to multiple point and put them in quod-tree for indexation
    
    mQT = new ElQT<cPMulTiepRed*,Pt2dr,cP2dGroundOfPMul>
              (
                     mPMul2Gr,
                     mBoxLoc,
                     5,  // 5 obj max / box
                     2*aSzTileAver
              );

    for (std::list<tMerge *>::const_iterator itM=mLMerge->begin() ; itM!=mLMerge->end() ; itM++)
    {
        cPMulTiepRed * aPM = new cPMulTiepRed(*itM,*this);
        if (mBoxLoc.inside(aPM->Pt()))
        {
           mLPMul.push_back(aPM);
           aVHist[(*itM)->NbSom()] ++;
           mQT->insert(aPM);
        }
        else
        {
           delete aPM;
        }
    }


    std::cout << "   NbMul " << mLMerge->size() 
              << " Nb2:" << aVHist[2] << " Nb3:" << aVHist[3] 
              << " Nb4:" << aVHist[4] << " Nb5:" << aVHist[5] 
              << " Nb6:" << aVHist[6] << "\n";
}




void cAppliTiepRed::GenerateSplit()
{
    ELISE_fp::MkDirSvp(mDir+TheNameTmp);

    Pt2dr aSzPix = mBoxGlob.sz() / mResol;
    Pt2di aNb = round_up(aSzPix / double(mSzTile));
    std::cout << "   GenerateSplit SzP=" << aSzPix << " Nb=" << aNb << " \n";
    Pt2dr aSzTile =  mBoxGlob.sz().dcbyc(Pt2dr(aNb));

    std::list<std::string> aLCom;


    int aCpt=0;
    for (int aKx=0 ; aKx<aNb.x ; aKx++)
    {
        for (int aKy=0 ; aKy<aNb.y ; aKy++)
        {
             Pt2dr aP0 = mBoxGlob._p0 + aSzTile.mcbyc(Pt2dr(aKx,aKy));
             Pt2dr aP1 = aP0 +aSzTile;
             Box2dr aBox(aP0,aP1);
             cElPolygone aPolyBox = cElPolygone::FromBox(aBox);
             // std::cout << "JJJJJjj " << aP0 << aP1 << "\n";
             cXml_ParamBoxReducTieP aParamBox;
             aParamBox.Box() = aBox;

             for (int aKC=0 ; aKC<int(mVecCam.size()) ; aKC++)
             {
                   // Polygone d'intersection
                   cElPolygone  aPolInter = aPolyBox  * mVecCam[aKC]->CS().EmpriseSol();
                   if (aPolInter.Surf() > 0)
                   {
               //          std::cout << "    SURF " << aPolInter.Surf() << " " << mVecCam[aKC]->NameIm() << "\n";
                        aParamBox.Ims().push_back(mVecCam[aKC]->NameIm());
                   }

             }
             if (aParamBox.Ims().size() >=2)
             {
                 MakeFileXML(aParamBox,NameParamBox(aCpt));
                 std::string aCom = GlobArcArgv + "  KBox=" + ToString(aCpt);
                 aLCom.push_back(aCom);
                 // std::cout << aCom << "\n";
                 aCpt++;
             }
        }
    }
    // cEl_GPAO::DoComInParal(aLCom);
    cEl_GPAO::DoComInSerie(aLCom);

    // int aNbX = round_up(mBoxGlob.sz().x /
}



void  cAppliTiepRed::Exe()
{
   if (mCallBack)
   {
      DoReduceBox();
   }
   else
   {
       GenerateSplit();
   }
}



int TestOscarTieP_main(int argc,char **argv) 
{
    cAppliTiepRed anAppli(argc,argv);
    anAppli.Exe();

    return EXIT_SUCCESS;
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
