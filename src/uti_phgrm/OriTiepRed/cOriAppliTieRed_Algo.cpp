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

// bool BUGTR = true;



/**********************************************************************/
/*                                                                    */
/*                         cAppliTiepRed                              */
/*                                                                    */
/**********************************************************************/




void cAppliTiepRed::DoLoadTiePoints(bool DoMaster)
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
               cCameraTiepRed & aCam2 = *(mMapCam[anI2]);
               // The result being symetric, the convention is that some data are stored only for  I1 < I2

               if (DoMaster)
               {
                   if (aCam1.IsMaster() || aCam2.IsMaster())
                   {
                       if (anI1 < anI2)
                          aCam1.LoadHomCam(aCam2);
                        else
                          aCam2.LoadHomCam(aCam1);
                   }
               }
               else  
               {
                   if ( (! aCam1.IsMaster()) && (!aCam2.IsMaster()))
                   {
                       ELISE_ASSERT(anI1<anI2,"Name Sort incAppliTiepRed::DoLoadTiePoints ");
                       aCam1.LoadHomCam(aCam2);
                   }
               }
            }
       }
   }
}

void cAppliTiepRed::DoLoadTiePoints()
{
    DoLoadTiePoints(true);
    DoLoadTiePoints(false);
}


void cAppliTiepRed::DoFilterCamAnLinks()
{
   // Select Cam ( and Link ) with enough points, and give a numeration to camera
   {
      // Suppress the links if one of its camera was supressed
      std::list<cLnk2ImTiepRed *> aNewL;
      for (std::list<cLnk2ImTiepRed *>::const_iterator itL=mLnk2Im.begin() ; itL!=mLnk2Im.end() ; itL++)
      {
          // if the two camera are to preserve, then preserve the link
          if ((*itL)->Cam1().SelectOnHom2Im() && (*itL)->Cam2().SelectOnHom2Im())
             aNewL.push_back(*itL);
      }
      // std::cout << "   LNK " << mLnk2Im.size() << " " << aNewL.size() << "\n";
      mLnk2Im = aNewL;

      std::vector<cCameraTiepRed *>  aNewV; // Filtered camera
      int aNum=0; // Current number
      for (int aKC=0 ; aKC<int(mVecCam.size()) ; aKC++)
      {
          if (mVecCam[aKC]->SelectOnHom2Im()) // If enouh point  camera is to preserve
          {
             aNewV.push_back(mVecCam[aKC]); // Add to new vec
             mVecCam[aKC]->SetNum(aNum);    // Give current numeration
             aNum++;
          }
          else
          {
             // Forget this camera
             mMapCam[mVecCam[aKC]->NameIm()] = 0;
             mVecCam[aKC]->SetDead();
             // delete mVecCam[aKC];
          }
      }
      // std::cout << "   CAMSS " << mVecCam.size() << " " << aNewV.size() << " NUM=" << aNum << "\n";
      mVecCam = aNewV; // Update member vector of cams

      mBufICam = std::vector<int>(mVecCam.size(),0);
      mBufICam2 = std::vector<int>(mVecCam.size(),0);


      std::vector<cLnk2ImTiepRed *> aVL0(mVecCam.size(),0);
      mVVLnk = std::vector<std::vector<cLnk2ImTiepRed *> >(mVecCam.size(),aVL0);
      for (std::list<cLnk2ImTiepRed *>::const_iterator itL = mLnk2Im.begin(); itL!=mLnk2Im.end() ; itL++)
      {
          cCameraTiepRed &  aC1 =   (*itL)->Cam1();
          cCameraTiepRed &  aC2 =   (*itL)->Cam2();
          if (aC1.Alive() && aC2.Alive())
          {
              ELISE_ASSERT(aC1.NameIm() < aC2.NameIm(),"order name image");
              mVVLnk[aC1.Num()][aC2.Num()] = *itL;
          }
      }


/*
      for (int aKC=0 ; aKC<int(mVecCam.size()) ; aKC++)
      {
           mVVLnk[aKC] = 
      }
*/
   }
}

void Verif(Pt2df aPf)
{
   Pt2dr aPd = ToPt2dr(aPf);
   if (std_isnan(aPd.x) || std_isnan(aPd.y))
   {
       std::cout << "PB PTS " << aPf << " => " << aPd << "\n";
       ELISE_ASSERT(false,"PB PTS in Verif");
   }
}


void cAppliTiepRed::DoReduceBox()
{


    DoLoadTiePoints();
    DoFilterCamAnLinks();
    // == OK

   // merge topological tie point

    // Create an empty merging struct
    mMergeStruct  = new  tMergeStr(mVecCam.size(),true);
    // for each link do the mergin
    for (std::list<cLnk2ImTiepRed *>::const_iterator itL=mLnk2Im.begin() ; itL!=mLnk2Im.end() ; itL++)
    {
        (*itL)->Add2Merge(mMergeStruct);
    }


    mMergeStruct->DoExport();                  // "Compile" to make the point usable


    mLMerge =  & mMergeStruct->ListMerged();    // Get the merged multiple points

    if (mModeIm) // Selectionne uniquement les  images connectees au master
    { 
        std::list<tMerge *> * aNewL = new std::list<tMerge *>;
        for (std::list<tMerge *>::const_iterator itM=mLMerge->begin() ; itM!=mLMerge->end() ; itM++)
        {
            if ((*itM)->IsInit(0))
            {
               aNewL->push_back(*itM);
            }
        }
        mLMerge  = aNewL;
    }
    std::vector<int> aVHist(mVecCam.size()+1,0);

    // Compute the average 
    double aSzTileAver = sqrt(mBoxLocQT.surf()/mLMerge->size()); 

    
    // Quod tree for spatial indexation
    mQT = new tTiePRed_QT ( mPMul2Gr, mBoxLocQT, 5  /* 5 obj max  box */, 2*aSzTileAver);
    // Heap for priority management


    if (mModeIm)
    {
        // std::cout << " Donne init QQQQ  " << aSzTileAver << " " << mBoxLocQT << "\n";
        // exit(EXIT_SUCCESS);
    }
    // OK
   // give ground coord to multiple point and put them in quod-tree  and  heap 
    {
       std::vector<double> aVPrec;
       mVPM.reserve(mLMerge->size());

       for (std::list<tMerge *>::const_iterator itM=mLMerge->begin() ; itM!=mLMerge->end() ; itM++)
       {
// std::cout << "AAAAA  " << mDistPMul << "  " <<  mResol   << "\n";
           cPMulTiepRed * aPM = new cPMulTiepRed(*itM,*this);
 // std::cout << "BBBBB  " << aPM->Pt() << "  \n";
           if (mBoxLocQT.inside(aPM->Pt()))
           {
              mVPM.push_back(aPM);
              
              aVPrec.push_back(aPM->Prec());

              ELISE_ASSERT((*itM)->NbSom()<int(aVHist.size()),"JJJJJJJJJJJJJJJJJJJJ");
              aVHist[(*itM)->NbSom()] ++;
              mQT->insert(aPM);
           }
           else
           {
              delete aPM;
           }
       }
//  Debut
       if (aVPrec.size() ==0)
       {   
          return;
       }
       mStdPrec = MedianeSup(aVPrec);  

       mHeap = new tTiePRed_Heap(mPMulCmp);
       // The gain can be computed once we know the standard precision
       for (int aKP=0 ; aKP<int(mVPM.size()) ; aKP++)
       {
           mVPM[aKP]->InitGain(*this);
           mHeap->push(mVPM[aKP]);
       }
    }
//  fin
    // int aNbInit = mHeap->nb();

    tPMulTiepRedPtr aPMPtr;
    while (mHeap->pop(aPMPtr))
    {

          mListSel.push_back(aPMPtr);
          aPMPtr->Remove();
          aPMPtr->SetSelected();
          std::set<tPMulTiepRedPtr>  aSetNeigh; // = *(new std::set<tPMulTiepRedPtr>);
          double aDist= mDistPMul * mResolQT;
          mQT->RVoisins(aSetNeigh,aPMPtr->Pt(),aDist);

          for (std::set<tPMulTiepRedPtr>::const_iterator itS=aSetNeigh.begin() ; itS!=aSetNeigh.end() ; itS++)
          {

              // tPMulTiepRedPtr aNeigh = aSetNeigh[aK];
              tPMulTiepRedPtr aNeigh = *itS;
              if (! aNeigh->Removed())
              {
                  aNeigh->UpdateNewSel(aPMPtr,*this);
                  if (aNeigh->Removable())
                  {
                      aNeigh->Remove();
                      mQT->remove(aNeigh);
                      mHeap->Sortir(aNeigh);
                  }
              }
          }
          /* std::cout << "         GAIN " << aPMPtr->Gain() << " " 
                       <<  aPMPtr->Merge()->NbArc()  << " " <<  aPMPtr->Merge()->NbSom() 
                       << " " << aSetNeigh.size() << " " <<  mHeap->nb() << "\n";
          */
          

    }

    int aNbInit = mListSel.size();
    VonGruber();

    // PAOK
    if (0)
    {
       std::cout << "NBPTS " <<  aNbInit << " Apr VonGruber => " <<  mListSel.size() << "\n";
    }
    DoExport();
}




void cAppliTiepRed::DoExport()
{
    int aNbNew=0;
    int aNbPrec=0;

    cStatArc aStatA;
    int aNbCam = mVecCam.size();
    std::vector<std::vector<ElPackHomologue> > aVVH (aNbCam,std::vector<ElPackHomologue>(aNbCam));
    for (std::list<tPMulTiepRedPtr>::const_iterator itP=mListSel.begin(); itP!=mListSel.end();  itP++)
    {
         tMerge * aMerge = (*itP)->Merge();
         const std::vector<Pt2di> &  aVE = aMerge->Edges();
         const std::vector<cCMT_U_INT1> &  aVA = aMerge->ValArc();

         aStatA.Add(aMerge->NbSom(),aMerge->NbArc());
         // std::cout << "SZZZZ " << aVE.size() << " " << aVA.size() << "\n";
         for (int aKCple=0 ; aKCple<int(aVE.size()) ; aKCple++)
         {
              if (aVA[aKCple].mVal!=ORR_MergePrec)
              {
                   aNbNew++;
                   int aKCam1 = aVE[aKCple].x;
                   int aKCam2 = aVE[aKCple].y;

                   const Pt2df & aP1 = aMerge->GetVal(aKCam1);
                   const Pt2df & aP2 = aMerge->GetVal(aKCam2);
                   cCameraTiepRed * aCam1 = mVecCam[aKCam1];
                   cCameraTiepRed * aCam2 = mVecCam[aKCam2];

                   if (mModeIm)
                   {
                       if (aCam1->NameIm() > aCam2->NameIm())
                       {
                           if (MPD_MM()) 
                           {
                              ELISE_ASSERT(false,"Incoherence in name ordering");
                           }
//  std::cout << "GGGGGgg " << ORR_MergeCompl << "\n";
                           ElSwap(aKCam1,aKCam2);
                           ElSwap(aCam1,aCam2);
                       }
                       cLnk2ImTiepRed * aLnK = LnkOfCams(aCam1,aCam2);
                       ELISE_ASSERT(aLnK!=0,"cLnk2ImTiepRed");
                       aLnK->VSelP1().push_back(aP1);
                       aLnK->VSelP2().push_back(aP2);
                       aLnK->VSelNb().push_back(2);
                       if (aVA[aKCple].mVal==ORR_MergeCompl)
                       {
                          // std::cout << "RRRRR_Compl=" << (*itP)->Residual(aKCam1,aKCam2,1000,*this) << "\n";
                           // Pt2dr aQ1 = aCam1->Hom2Cam(aP1);
                           // Pt2dr aQ2 = aCam2->Hom2Cam(aP2);
                       }
                   }
                   else
                   {

                       Pt2dr aQ1 = aCam1->Hom2Cam(aP1);
                       Pt2dr aQ2 = aCam2->Hom2Cam(aP2);

                       aVVH[aKCam1][aKCam2].Cple_Add(ElCplePtsHomologues(aQ1,aQ2));
                       aVVH[aKCam2][aKCam1].Cple_Add(ElCplePtsHomologues(aQ2,aQ1));

                       if (VerifNM())
                       {
                       // Pt2dr aW1 = mNM->CalibrationCamera(aCam1->NameIm());
                       // std::cout << "FFFFffGG  :" << mNM->CalibrationCamera(aCam1->NameIm())->Radian2Pixel(Pt2dr(aP1.x,aP1.y)) - aQ1 << "\n";
                       }
               

                       Verif(aP1);
                       Verif(aP2);
                   }
              }
              else
              {
                   aNbPrec++;
              }
         }
    }

    if (mModeIm)
    {
       for (std::list<cLnk2ImTiepRed *>::const_iterator itL=mLnk2Im.begin() ; itL!=mLnk2Im.end() ; itL++)
       {
           mNM->WriteCouple
           (
               NameHomol((*itL)->Cam1().NameIm(),(*itL)->Cam2().NameIm(),mKBox),
               (*itL)->VSelP1(),
               (*itL)->VSelP2(),
               (*itL)->VSelNb()
           );
       }   

       if (0)
       {
          std::cout << " Master=" << mMasterIm << "\n";
          std::cout << " RatioNew " << aNbNew / double(aNbNew+aNbPrec) << "\n";
          aStatA.Show();
          getchar();
       }
    }
    else
    {
        int aSomH=0;
        for (int aKCam1=0 ; aKCam1<aNbCam ; aKCam1++)
        {
            for (int aKCam2=0 ; aKCam2<aNbCam ; aKCam2++)
            {
                 const ElPackHomologue & aPack = aVVH[aKCam1][aKCam2];
                 aSomH += aPack.size();
                 if (aPack.size())
                 {
                      aPack.StdPutInFile(NameHomol(mVecCam[aKCam1]->NameIm(),mVecCam[aKCam2]->NameIm(),mKBox));
                 }
            }
        }
    }
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
