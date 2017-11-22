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

#include "MergeCloud.h"

void  cAppliMergeCloud::CreateGrapheConx()
{
   mRecouvrTot  =  mVSoms.size();  // Recouvrement total
            // Couple d'homologues
   std::string aKSH = "NKS-Set-Homol@" + mSH + "@" + mParam.ExtHom().Val();
   std::string aKAH = "NKS-Assoc-CplIm2Hom@" + mSH + "@" + mParam.ExtHom().Val();

   std::vector<tMCArc *> aVAddCur;
   const cInterfChantierNameManipulateur::tSet * aSetHom = ICNM()->Get(aKSH);




   for (int aKH=0 ; aKH<int(aSetHom->size()) ; aKH++)
   {
       const std::string & aNameFile = (*aSetHom)[aKH];
       std::string aFullNF = Dir() + aNameFile;
       if (sizeofile(aFullNF.c_str()) > mParam.MinSzFilHom().Val())
       {
           std::pair<std::string,std::string> aPair = ICNM()->Assoc2To1(aKAH,aNameFile,false);
           tMCSom * aS1 = SomOfName(aPair.first);
           tMCSom * aS2 = SomOfName(aPair.second);
           // if ((aS1!=0) && (aS2!=0) && (sizeofile(aNameFile.c_str())>mParam.MinSzFilHom().Val()))
           // MPD : redondant + erreur car pas aFullNF 
           if ((aS1!=0) && (aS2!=0) )
           {
              tMCArc *  anArc = TestAddNewarc(aS1,aS2);
              if (anArc)
                 aVAddCur.push_back(anArc);
           }
       }
   }
            // Ajout recursif des voisin
   while (! aVAddCur.empty())
   {
       std::cout << "ADD " << aVAddCur.size() << "\n";
       std::vector<tMCArc *> aVAddNew;
       for (int aK=0 ; aK<int(aVAddCur.size()) ; aK++)
       {
           tMCArc & anArc = *(aVAddCur[aK]);
           AddVoisVois(aVAddNew,anArc.s1(),anArc.s2());
           AddVoisVois(aVAddNew,anArc.s2(),anArc.s1());
       }
       aVAddCur = aVAddNew;
   }

   // Calcul des voisins proches (contenu dans Apero ChImSec)
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
       tMCSom * aS1 =  mVSoms[aK];

       const cOneSolImageSec *   aSol = aS1->attr()->SolOfCostPerIm(mParam.CostPerImISOM().Val());

       if (aSol)
       {
           for 
           (
               std::list<std::string>::const_iterator itS=aSol->Images().begin();
               itS !=aSol->Images().end();
               itS++
           )
           {
                tMCSom *  aS2 = SomOfName(*itS);
                if (aS2)
                {
                    tMCArc * anArc = mGr.arc_s1s2(*aS1,*aS2);
                    if (anArc)
                    {
                       // anArc->sym_flag_set_kth_true(mFlagCloseN);
                       anArc->flag_set_kth_true(mFlagCloseN);
                       aS1->attr()->AddCloseVois(aS2->attr());
                    }
                    // std::cout << "AAAaA  :" << anArc << "\n";
                }
           }
       }
   }

   // Na pas effacer, permet de voir le graphe des close image en cas de doute
   if (1)
   {
       for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
       {
           tMCSom * aS1 =  mVSoms[aK];
           std::cout << aS1->attr()->IMM()->mNameIm 
                     << " All: " << aS1->nb_succ(mSubGrAll) 
                     << " Closed: " << aS1->nb_succ(mSubGrCloseN) ;

            std::cout << " ";
            for (tArcIter itA = aS1->begin(mSubGrCloseN) ; itA.go_on() ; itA++)
            {
                const std::string & aN2 = (*itA).s2().attr()->IMM()->mNameIm;
                std::cout << ExtractDigit(StdPrefixGen(aN2),"0000") << " ";

            }
            std::cout << "\n";
       }
   }

   mRecMoy = mRecouvrTot /  mVSoms.size();
   mNbImMoy = mVSoms.size() / mRecMoy;  // En Fait NbIm^2 /mRecouvrTot
   std::cout  << "REC TOT " << mRecouvrTot << " RMoy " << mRecMoy << " NbIm " << mNbImMoy << "\n";
}





void  cAppliMergeCloud::AddVoisVois(std::vector<tMCArc *> & aVArc,tMCSom& aS1,tMCSom& aS2)
{
    for (tArcIter itA = aS2.begin(mSubGrAll) ; itA.go_on() ; itA++)
    {
       tMCArc * anArc = TestAddNewarc(&aS1,&(itA->s2()));
       if (anArc) 
          aVArc.push_back(anArc);
    }
}

tMCArc * cAppliMergeCloud::TestAddNewarc(tMCSom * aS1,tMCSom *aS2)
{
   if (aS1 == aS2) return 0;
   if (aS1 > aS2) ElSwap(aS1,aS2);

   tMCPairS aPair(aS1,aS2);

   if (mTestedPairs.find(aPair) != mTestedPairs.end())
      return 0; // Deja fait
   mTestedPairs.insert(aPair) ;  // plus a faire

   cASAMG * anA1 = aS1->attr() ;
   cASAMG * anA2 = aS2->attr() ;

   double aR1On2 = anA1->LowRecouvrt(*anA2);
   double aR2On1 = anA2->LowRecouvrt(*anA1);

   mRecouvrTot += aR1On2 + aR2On1;

   if ((aR1On2< mParam.TauxRecMin()) && (aR2On1 <mParam.TauxRecMin()))
      return 0;


   if (0)
   {
       std::cout << "AddArc: " << anA1->IMM()->mNameIm << " " 
                               << anA2->IMM()->mNameIm 
                               << " Rec: " << aR1On2 << "/" << aR2On1 << "\n";
   }
   ELISE_ASSERT
   (
      mGr.arc_s1s2(*aS1,*aS2)==0,
      "Incoherence in cAppliMergeCloud::TestAddNewarc"
   );

   c3AMGS *  anAs = new c3AMGS;
   c3AMG  * anA12 = new c3AMG(anAs,aR1On2);
   c3AMG  * anA21 = new c3AMG(anAs,aR2On1);
   
   tMCArc & anArc = mGr.add_arc(*aS1,*aS2,anA12,anA21);

   return & anArc;
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
