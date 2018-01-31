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


#include "NewRechPH.h"

class cAppliStatPHom;
class cOneImSPH;



class cAppliStatPHom
{
    public :
       friend class cOneImSPH;
       cAppliStatPHom(int argc,char ** argv);

       bool I1HasHom(const Pt2dr & aP1) {return mNuage1->CaptHasData(mNuage1->Plani2Index(aP1));}
       Pt2dr Hom(const Pt2dr & aP1);


    private :
       void ShowStat(const std::string & aMes,int aNb,std::vector<double>  aVR,double aPropMax=1.0);
       void TestHom();
       std::string mDir;
       cInterfChantierNameManipulateur * mICNM;
       cOneImSPH * mI1;
       cOneImSPH * mI2;
       std::string mOri;
       std::string mSH;
       std::string mNameNuage;

       cElNuage3DMaille * mNuage1;
       
       ElPackHomologue   mPack;
       bool              mSetPI;
       std::string mExtInput;

       double EcartEpip(const Pt2dr & aP1,const Pt2dr & aP2);
       double EcartCompl(const Pt2dr & aP1,const Pt2dr & aP2);
};

class cOneImSPH
{
    public :
         cOneImSPH(const std::string & aName,cAppliStatPHom & anAppli) ;
         void TestMatch(cOneImSPH & aI2);

         cOnePCarac * Nearest(const Pt2dr&,int aKl,double & aDMin,double aMinDMin);
             

         cAppliStatPHom &   mAppli;
         std::string        mN;
         cBasicGeomCap3D *  mCam;
         cSetPCarac *       mSPC;
         std::vector<std::vector<cOnePCarac*> > mVVPC; // Classe par label
         std::vector<cOnePCarac*>               mVNearest;
         std::vector<std::vector<cOnePCarac*> > mVHom;
         Tiff_Im            mTif;
};


/*******************************************************/
/*                                                     */
/*           cOneImSPH                                 */
/*                                                     */
/*******************************************************/

cOneImSPH::cOneImSPH(const std::string & aName,cAppliStatPHom & anAppli) :
   mAppli   (anAppli),
   mN       (aName),
   mCam     (mAppli.mICNM->StdCamGenerikOfNames(mAppli.mOri,mN)),
   mSPC     (LoadStdSetCarac(mN,anAppli.mExtInput)),
   mVVPC    (int(eTPR_NoLabel)),
   mTif     (Tiff_Im::StdConvGen(aName,1,true))
{
   for (auto & aPt : mSPC->OnePCarac())
   {
      if (mAppli.mSetPI) 
      {
         aPt.Pt() = aPt.Pt0();
      }
      mVVPC.at(int(aPt.Kind())).push_back(&aPt);
   }
   std::cout << " N=" << mN << " nb=" << mSPC->OnePCarac().size() << "\n";
}


cOnePCarac * cOneImSPH::Nearest(const Pt2dr& aP0,int aKLab,double &aDMin,double aMinDMin)
{
    aDMin = 1e10;
    cOnePCarac * aRes = nullptr;
    for (auto & aPt : mVVPC[aKLab])
    {
        double aD = euclid(aP0-aPt->Pt());
        if ((aD<aDMin) && (aD>aMinDMin))
        {
            aDMin = aD;
            aRes = aPt;
        }
    }
    ELISE_ASSERT(aRes!=0,"cOneImSPH::Nearest");
    return aRes;
}

void cOneImSPH::TestMatch(cOneImSPH & aI2)
{

   for (int aKL=0 ; aKL<int(eTPR_NoLabel) ; aKL++)
   {
        mVHom.push_back(std::vector<cOnePCarac*>());
        int aDifMax = 3;
        std::vector<int>  aHistoScale(aDifMax+1,0);
        double aSeuilDist = 2.0;
        double aSeuilProp = 0.02;
        int aNbOk=0;

        eTypePtRemark aLab = eTypePtRemark(aKL);
        const std::vector<cOnePCarac*>  &   aV1 = mVVPC[aKL];
        const std::vector<cOnePCarac*>  &   aV2 = aI2.mVVPC[aKL];

        std::vector<cOnePCarac>  aVObj1;
        for (auto aPtr1 : aV1)
            aVObj1.push_back(*aPtr1);


        if ((!aV1.empty()) && (!aV2.empty()))
        {
            aI2.mVNearest.clear();
            std::cout << "===========================================================\n";
            std::cout << "For " << eToString(aLab) << " sz=" << aV1.size() << " " << aV2.size() << "\n";

            std::vector<double> aVD22;
            for (int aK2=0 ; aK2< int(aV2.size()); aK2++)
            {
                 double aDist;
                 cOnePCarac * aP = aI2.Nearest(aV2[aK2]->Pt(),aKL,aDist,1e-5);
                 aI2.mVNearest.push_back(aP);
                 aVD22.push_back(aDist);
            }
            mAppli.ShowStat("Nearest D for ",20,aVD22);
      
 
            std::vector<double> aVD12;
            std::vector<double> aScorInvR;
            for (int aK1=0 ; aK1< int(aV1.size()); aK1++)
            {
                Pt2dr aP1 = aV1[aK1]->Pt();
                cOnePCarac * aHom = 0;
                if (mAppli.I1HasHom(aP1))
                {
                    double aDist;
                    cOnePCarac * aP = aI2.Nearest(mAppli.Hom(aP1),aKL,aDist,0.0);
                    aVD12.push_back(aDist);
                    if (aDist<aSeuilDist)
                    {
                         aNbOk++;
                         aHistoScale.at(ElMin(aDifMax,ElAbs(aV1[aK1]->NivScale() - aP->NivScale())))++;
                         double aPropInv = 1 - ScoreTestMatchInvRad(aVObj1,aV1[aK1],aP);
                         aScorInvR.push_back(aPropInv);
                         // if (aNbOk%10) std::cout << "aNbOk++aNbOk++ " << aNbOk << "\n";
                         if (aPropInv < aSeuilProp)
                            aHom = aP;
                    }
                }

                mVHom.at(aKL).push_back(aHom);
            }

            mAppli.ShowStat("By Homol D for ",20,aVD12,0.5);
            mAppli.ShowStat("Inv Rad ",20,aScorInvR,1.0);

            for (int aK=0 ; aK<=aDifMax ; aK++)
            {
               int aNb = aHistoScale.at(aK);
               if (aNb)
                  std::cout << "  * For dif="  << aK << " perc=" << (aNb * 100.0) / aNbOk << "\n";
            }

            getchar();
        }
   }
}


/*******************************************************/
/*                                                     */
/*           cAppliStatPHom                            */
/*                                                     */
/*******************************************************/


Pt2dr cAppliStatPHom::Hom(const Pt2dr & aP1)
{
   ELISE_ASSERT(I1HasHom(aP1),"cAppliStatPHom::Hom");
   Pt2dr aI1 = mNuage1->Plani2Index(aP1);
   Pt3dr aPTer = mNuage1->PtOfIndexInterpol(aI1);
   return mI2->mCam->Ter2Capteur(aPTer);
}

void  cAppliStatPHom::ShowStat(const std::string & aMes,int aNB,std::vector<double>  aVR,double aPropMax)
{
    if (aVR.empty())
       return;

    std::cout << "=========  " << aMes << " ==========\n";
    for (int aK=0 ; aK< aNB ; aK++)
    {
        double aProp= ((aK+0.5) / double(aNB)) * aPropMax;
        std::cout << "E[" << round_ni(100.0*aProp) << "]= " << KthValProp(aVR,aProp) << "\n";
    }
    double aSom = 0.0;
    for (const auto & aV : aVR)
    {
        aSom += aV;
    }
    std::cout << "   MOY= " << aSom/aVR.size() << "\n";
}



double cAppliStatPHom::EcartEpip(const Pt2dr & aP1,const Pt2dr & aP2)
{
    return  mI1->mCam->EpipolarEcart(aP1,*mI2->mCam,aP2);
}


double cAppliStatPHom::EcartCompl(const Pt2dr & aP1,const Pt2dr & aP2)
{
    ELISE_ASSERT(mNuage1!=0,"cAppliStatPHom::EcartCompl");

    if (! I1HasHom(aP1)) return -1;
    return euclid(aP2-Hom(aP1));
}

void cAppliStatPHom::TestHom()
{
    std::vector<double> aVREpi;
    std::vector<double> aVRComp;
    for (cPackNupletsHom::iterator itP=mPack.begin() ; itP!=mPack.end() ; itP++)
    {
        double anEcartEpi = EcartEpip(itP->P1(),itP->P2());
        aVREpi.push_back(ElAbs(anEcartEpi));
        if (mNuage1)
        {
           double anEcartCompl = EcartCompl(itP->P1(),itP->P2());
           if (anEcartCompl>=0)
           {
               aVRComp.push_back(anEcartCompl);
           }
        }
    }
    ShowStat("ECAR EPIP",20,aVREpi);
    ShowStat("ECAR COMPL",20,aVRComp);
/*
    int aNB= 20;
    std::cout << "========= ECAR EPIP ==========\n";
    for (int aK=0 ; aK< aNB ; aK++)
    {
        double aProp= (aK+0.5) / double(aNB);
        std::cout << "E[" << round_ni(100.0*aProp) << "]= " << KthValProp(aVR,aProp) << "\n";
    }
*/
}


cAppliStatPHom::cAppliStatPHom(int argc,char ** argv) :
    mDir ("./"),
    mSH  (""),
    mNuage1  (0),
    mSetPI   (false),
    mExtInput ("Std")
{
   std::string aN1,aN2;
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aN1, "Name Image1")
                     << EAMC(aN2, "Name Image2")
                     << EAMC(mOri,"Orientation"),
         LArgMain()  << EAM(mSH,"SH",true,"Set of homologous point")
                     << EAM(mNameNuage,"NC",true,"Name of cloud")
                     << EAM(mSetPI,"SetPI",true,"Set Integer point, def=false,for stat")
                     << EAM(mExtInput,"ExtInput",true,"Extentsion for tieP")
   );

   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   StdCorrecNameOrient(mOri,mDir);
   StdCorrecNameHomol(mSH,mDir);
   mI1 = new cOneImSPH(aN1,*this);

/*
   if (1)
   {
       for (int aNbB=1 ; aNbB<4 ; aNbB++)
       {
          cFullParamCB  aFB = RandomFullParamCB(mI1->mSPC->OnePCarac()[0],aNbB,3);
          MakeFileXML(aFB,"Test_"+ToString(aNbB)+".xml");
       }
   }
*/
   mI2 = new cOneImSPH(aN2,*this);

   if (EAMIsInit(&mNameNuage))
      mNuage1 = cElNuage3DMaille::FromFileIm(mNameNuage);

   mPack = mICNM->StdPackHomol(mSH,mI1->mN,mI2->mN);
   TestHom();


   mI1->TestMatch(*mI2);
}


int  CPP_StatPHom(int argc,char ** argv)
{
    cAppliStatPHom anAppli(argc,argv);

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
aooter-MicMac-eLiSe-25/06/2007*/
