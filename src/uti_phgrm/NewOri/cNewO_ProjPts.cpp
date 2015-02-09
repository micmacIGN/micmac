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


/**************************************************************/
/*                                                            */
/*           cProjCple                                        */
/*                                                            */
/**************************************************************/

cProjCple::cProjCple(const Pt3dr & aP1,const Pt3dr & aP2,double aPds) :
    mP1  (aP1),
    mP2  (aP2),
    mPds (aPds)
{
}

const Pt3dr & cProjCple::P1() const {return mP1;}
const Pt3dr & cProjCple::P2() const {return mP2;}

cProjCple cProjCple::Spherik(const ElCamera & aCam1,const Pt2dr & aP1,const ElCamera & aCam2,const Pt2dr &aP2,double aPds)
{
    Pt3dr aQ1 =  aCam1.F2toDirRayonL3(aP1);
    Pt3dr aQ2 =  aCam2.F2toDirRayonL3(aP2);

    return cProjCple(vunit(aQ1),vunit(aQ2),aPds);
}

static Pt3dr Proj(const Pt3dr & aP) {return Pt3dr(aP.x/aP.z,aP.y/aP.z,1.0);}

cProjCple cProjCple::Projection(const ElCamera & aCam1,const Pt2dr & aP1,const ElCamera & aCam2,const Pt2dr &aP2,double aPds)
{
    Pt3dr aQ1 =  aCam1.F2toDirRayonL3(aP1);
    Pt3dr aQ2 =  aCam2.F2toDirRayonL3(aP2);

    return cProjCple(Proj(aQ1),Proj(aQ2),aPds);
}

/**************************************************************/
/*                                                            */
/*                  cProjListHom                              */
/*                                                            */
/**************************************************************/



template <const int TheNbPts,class Type>  class cFixedMergeTieP
{
     public :
       typedef cFixedMergeTieP<TheNbPts,Type> tMerge;
       typedef std::map<Type,tMerge *>     tMapMerge;


       cFixedMergeTieP() :
           mOk     (true),
           mNbArc  (0)
       {
           for (int aK=0 ; aK<TheNbPts; aK++)
           {
               mTabIsInit[aK] = false;
           }
       }


       void FusionneInThis(cFixedMergeTieP<TheNbPts,Type> & anEl2,tMapMerge * Tabs)
       {
            if ((!mOk) || (! anEl2.mOk))
            {
                mOk = anEl2.mOk = false;
                return;
            }
            mNbArc += anEl2.mNbArc;
            for (int aK=0 ; aK<TheNbPts; aK++)
            {
                if ( mTabIsInit[aK] && anEl2.mTabIsInit[aK] )
                {
                   // Ce cas ne devrait pas se produire, il doivent avoir ete fusionnes
                   ELISE_ASSERT(false,"cFixedMergeTieP");
                }
                else if ( (!mTabIsInit[aK]) && anEl2.mTabIsInit[aK] )
                {
                     mVals[aK] = anEl2.mVals[aK] ;
                     mTabIsInit[aK] = true;
                     Tabs[aK][mVals[aK]] = this;
                }
            }
       }
       void AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2)
       {
           AddSom(aV1,aK1);
           AddSom(aV2,aK2);
           mNbArc ++;
       }

        bool IsInit(int aK) const {return mTabIsInit[aK];}
        const Type & GetVal(int aK)    const {return mVals[aK];}
        bool IsOk() const {return mOk;}
        void SetNoOk() {mOk=false;}
        int  NbArc() const {return mNbArc;}
        void IncrArc() { mNbArc++;}
       
     private :
        void AddSom(const Type & aV,int aK)
        {
           if (mTabIsInit[aK])
           {
               if (mVals[aK] != aV)
               {
                   mOk = false;
               }
           }
           else 
           {
               mVals[aK] = aV;
               mTabIsInit[aK] = true;
           }
        }
        Type mVals[TheNbPts];
        bool  mTabIsInit[TheNbPts];
        bool  mOk;
        int   mNbArc;
};

cFixedMergeTieP<2,Pt2dr> anEl2;
cFixedMergeTieP<3,Pt2dr> anEl3;

template <const int TheNb,class Type> class cFixedMergeStruct
{
     public :
        typedef cFixedMergeTieP<TheNb,Type> tMerge;
        typedef std::map<Type,tMerge *>     tMapMerge;
        typedef typename tMapMerge::iterator         tItMM;

        std::list<tMerge *> Export()
        {
              std::list<tMerge *> aRes;
              for (int aK=0 ; aK<TheNb ; aK++)
              {
                  tMapMerge & aMap = mTheMaps[aK];
                  for (tItMM anIt = aMap.begin() ; anIt != aMap.end() ; anIt++)
                  {
                      tMerge * aM = anIt->second;
                      if (aM->IsOk())
                      {
                          aRes.push_back(aM);
                          aM->SetNoOk();
                      }
                  }
              }
              return aRes;
        }

        void AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2)
        {
             tMapMerge & aMap1 = mTheMaps[aK1];
             tItMM anIt1  = mTheMaps[aK1].find(aV1);
             tMerge * aM1 = (anIt1 != aMap1.end()) ? anIt1->second : 0;

             tMapMerge & aMap2 = mTheMaps[aK2];
             tItMM anIt2  = mTheMaps[aK2].find(aV2);
             tMerge * aM2 =  (anIt2 != aMap2.end()) ? anIt2->second : 0;
             tMerge * aMerge = 0;

             if ((aM1==0) && (aM2==0))
             {
                 aMerge = new tMerge;
                 aMap1[aV1] = aMerge;
                 aMap2[aV2] = aMerge;
             }
             else if ((aM1!=0) && (aM2!=0))
             {
                  if (aM1==aM2) 
                  {   
                     aM1->IncrArc();
                     return;
                  }
                  aM1->FusionneInThis(*aM2,mTheMaps);
                  if (aM1->IsOk() && aM2->IsOk())
                  {
                     delete aM2;
                     aMerge = aM1;
                  }
                  else
                     return;
             }
             else if ((aM1==0) && (aM2!=0))
             {
                 aMerge = mTheMaps[aK1][aV1] = aM2;
             }
             else
             {
                 aMerge =  mTheMaps[aK2][aV2] = aM1;
             }
             aMerge->AddArc(aV1,aK1,aV2,aK2);
        }

     private :
        tMapMerge                           mTheMaps[TheNb];
};


cFixedMergeStruct<2,Pt2dr> aMap2;

template <const int TheNb> void AddPackHom
                           (
                                cFixedMergeStruct<TheNb,Pt2dr> & aMap,
                                const ElPackHomologue & aPack,
                                const ElCamera & aCam1,int aK1,
                                const ElCamera & aCam2,int aK2
                           )
{
    for 
    (
          ElPackHomologue::tCstIter itH=aPack.begin();
          itH !=aPack.end();
          itH++
    )
    {
         ElCplePtsHomologues aCple = itH->ToCple();
         Pt2dr aP1 =  ProjStenope(aCam1.F2toDirRayonL3(aCple.P1()));
         Pt2dr aP2 =  ProjStenope(aCam2.F2toDirRayonL3(aCple.P2()));
         aMap.AddArc(aP1,aK1,aP2,aK2);
    }
}


/**********************************************************************************************/
/*                                                                                            */
/*                           BENCHS                                                           */
/*                                                                                            */
/**********************************************************************************************/

std::vector<int> CptArc(const std::list<cFixedMergeTieP<2,Pt2dr> *> aL,int & aTot)
{
    std::vector<int> aRes(100);
    aTot = 0;

    for (std::list<cFixedMergeTieP<2,Pt2dr> *>::const_iterator  itR=aL.begin() ; itR!=aL.end() ; itR++)
    {
        int aNb = (*itR)->NbArc();
        aTot += aNb;
        aRes[aNb] ++;
    }
     return aRes;
}

void AssertCptArc(cFixedMergeStruct<2,Pt2dr> & aMap ,int aNb0,int aNb1,int aNb2)
{
   std::list<cFixedMergeTieP<2,Pt2dr> *>  aRes = aMap.Export();
   int aTot;
   std::vector<int> aCpt = CptArc(aRes,aTot);

   if ((aNb0 != aCpt[0]) || (aNb1 != aCpt[1])   || (aNb2 != aCpt[2]) )
   {
      std::cout << "Nb0 " << aCpt[0] << " ; Nb1=" << aCpt[1] << " ; Nb2=" << aCpt[2] << "\n";
      ELISE_ASSERT(false,"AssertCptArc");
   }
}


void  NO_MergeTO_Test2_Basic(int aK)
{
    cFixedMergeStruct<2,Pt2dr> aMap;
    
    aMap.AddArc(Pt2dr(0,0),0,Pt2dr(1,1),1);

    if (aK==0)
    {
       AssertCptArc(aMap,0,1,0);  // 1 ARc
       return ;
    }

    aMap.AddArc(Pt2dr(1,1),1,Pt2dr(0,0),0);  
    if (aK==1)
    {
       AssertCptArc(aMap,0,0,1); // Arc Sym
       return ;
    }

    aMap.AddArc(Pt2dr(2,2),0,Pt2dr(3,3),1);
    if (aK==2)
    {
       AssertCptArc(aMap,0,1,1);  // 1 Sym + 1 ASym
       return ;
    }

    aMap.AddArc(Pt2dr(2,2),0,Pt2dr(4,4),1);  // Incoherent
    if (aK==3)
    {
       AssertCptArc(aMap,0,0,1);  //  1 Sym, 1 Inco
       return ;
    }
    return ;
}




void  NO_MergeTO_Test2_0
(  
      const ElCamera & aCam1,
      const ElPackHomologue & aPack12,
      const ElCamera & aCam2,const ElPackHomologue & aPack21
)  
{
    cFixedMergeStruct<2,Pt2dr> aMap2;
    AddPackHom(aMap2,aPack12,aCam1,0,aCam2,1);
    AddPackHom(aMap2,aPack21,aCam2,1,aCam1,0);
    std::list<cFixedMergeTieP<2,Pt2dr> *>  aRes = aMap2.Export();
    int aNb1=0;
    int aNb2=0;
    for (std::list<cFixedMergeTieP<2,Pt2dr> *>::const_iterator  itR=aRes.begin() ; itR!=aRes.end() ; itR++)
    {
        if ((*itR)->NbArc() ==1)
        {
           aNb1++;
        }
        else if ((*itR)->NbArc() ==2)
        {
           aNb2++;
        }
        else 
        {
           ELISE_ASSERT(false,"NO_MergeTO_Test2_0");
        }
    }
    std::cout << "INPUT " << aPack12.size() << " " << aPack21.size() << " Exp " << aNb1 << " " << aNb2 << "\n";
}



void  NO_MergeTO_Test4_Basic()
{
    cFixedMergeStruct<4,Pt2dr> aMap;
    
    aMap.AddArc(Pt2dr(0,0),0,Pt2dr(1,1),1);
    aMap.AddArc(Pt2dr(2,2),2,Pt2dr(3,3),3);

    aMap.AddArc(Pt2dr(2,2),2,Pt2dr(1,1),1);

    std::list<cFixedMergeTieP<4,Pt2dr> *>  aRes = aMap.Export();

    ELISE_ASSERT(aRes.size()==1,"NO_MergeTO_Test4_Basic");
    ELISE_ASSERT((*aRes.begin())->NbArc()==3,"NO_MergeTO_Test4_Basic");
}



int TestNewOriImage_main(int argc,char ** argv)
{
   for (int aK=0 ; aK< 10 ; aK++)
   {
     NO_MergeTO_Test2_Basic(aK);
   }
   NO_MergeTO_Test4_Basic();
/*
   std::string aNameOri,aNameI1,aNameI2;
   std::vector<std::string> aVNames;


   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(aNameI1,"Name First Image")
                   <<  EAMC(aNameI2,"Name Second Image"),
        LArgMain() << EAM(aNameOri,"Ori",true,"Orientation ")
   );

    cNewO_NameManager aNM("./",aNameOri,"dat");

    aVNames.push_back(aNameI1);
    aVNames.push_back(aNameI2);

    
    std::vector<CamStenope *> aVC;

    for (int aK=0 ; aK<int(aVNames.size()) ; aK++)
    {
         aVC.push_back(aNM.CamOfName(aVNames[aK]));
    }


    if (aVNames.size()==2)
    {
        if (1)
        {
            ElPackHomologue aLH12 = aNM.PackOfName(aVNames[0],aVNames[1]);
            ElPackHomologue aLH21 = aNM.PackOfName(aVNames[1],aVNames[0]);
            NO_MergeTO_Test2_0(*(aVC[0]),aLH12,*(aVC[1]),aLH21);
        }

        
    }

*/
/*
    ElPackHomologue aLH = aNM.PackOfName(aNameI1,aNameI2);
    std::cout << "FFF " << aC1->Focale() << " " << aC2->Focale() << " NBh : " << aLH.size() << "\n";
*/

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
