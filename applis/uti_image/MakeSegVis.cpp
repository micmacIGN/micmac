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
     Image SAR sur /mnt/saim_containers/image_saim/

           255  256
           
           335

           362  365 (?)
*/

#include "api/vecto.h"
#include "general/all.h"
#include "private/all.h"
#include "im_special/hough.h"     


/*******************************************************************/
/*******************************************************************/


class TestHMI : public  HoughMapedInteractor
{
     public :
         virtual void OnNewSeg(const ComplexR &,const ComplexR &);
         virtual void OnNewCase(const ComplexI &,const ComplexI &);

         TestHMI (Video_Win * aW,Pt2di aP0);

     private :
         Video_Win  * mW;
         Pt2di   mP0;
};



TestHMI::TestHMI(Video_Win  * aW,Pt2di aP0) :
    HoughMapedInteractor (),
    mW                   (aW),
    mP0                  (aP0)
{
}

void TestHMI::OnNewSeg(const ComplexR & p0,const ComplexR & p1)
{
   if (mW)
   {
      mW->draw_seg 
      (
         Std2Elise(p0)-mP0,
         Std2Elise(p1)-mP0,
         mW->pdisc()(P8COL::red)
      );
   }
}

void TestHMI::OnNewCase(const  ComplexI & StdP0,const ComplexI & StdP1)
{
   Pt2di p0 = Std2Elise(StdP0);
   Pt2di p1 = Std2Elise(StdP1);
   if (mW)
   {
      mW->draw_rect ( p0-mP0, p1-mP0, mW->pdisc()(P8COL::green));
   }
   else
   {
      cout << "Done Rect " << p0 << p1 << "\n";
   }
}


bool APPLI_MODE_SAR = false;



class SarHoughAppli
{
     public :

        SarHoughAppli
        (
            Tiff_Im  aTif,
            std::string,
            Pt2di SzIm
        );

		virtual ~SarHoughAppli(){};

        void load(Pt2di P0File,HoughMapedParam &,const std::string &);

        void MakeHough(Pt2di p0Im);


        virtual Video_Win * W() {return 0;}

     protected :

           virtual Output GrayVisuIm() {return Output::onul();}

           Fonc_Num               mImFile;
           std::string            mNameIm;
           Pt2di                  mSzIm;
           Pt2di                  mSzH;

           vector<Seg2d>          mVSegLoc;
           vector<Seg2d>          mVSegGlob;

           virtual void draw_segWIm(const vector<Seg2d> &,INT){}
};






void SarHoughAppli::load
     (
          Pt2di P0File,
          HoughMapedParam & aParam, 
          const std::string &  aNameSeg
     )
{

   Flux_Pts AllPts = rectangle(Pt2di(0,0),mSzIm);

    Fonc_Num tr_ImInit = trans(mImFile,P0File);
    if (W())
       ELISE_COPY(AllPts,mod(tr_ImInit,256),GrayVisuIm());
       //ELISE_COPY(AllPts,Max(0,Min(255,tr_ImInit)),GrayVisuIm());



    TestHMI anHInteractor(W(),P0File);


    HoughMapFromFile
    (
       mNameIm,
       Elise2Std(P0File),
       Elise2Std(P0File+mSzIm),
       aParam,
       anHInteractor
    );
    getchar();
    cout << "END HOUGH \n";

   
    list<Seg2d> lSeg;

    for (INT k=0; k<2 ; k++)
    {
       ELISE_COPY
       (
           AllPts,
           (k==0)?tr_ImInit:Fonc_Num(255),
           GrayVisuIm()
       );

       const std::vector<ComplexR> VP0 = anHInteractor.Extr0();
       const std::vector<ComplexR> VP1 = anHInteractor.Extr1();
       for
       (
          std::vector<ComplexR>::const_iterator it0 =VP0.begin(),it1=VP1.begin();
          it0 != VP0.end();
          it0++,it1++
       )
       {
           Seg2d aSeg(Std2Elise(*it0),Std2Elise(*it1));
           if (k==0)
              lSeg.push_back(aSeg);
           if (W())
           {
              ELISE_COPY
              (
                  line(aSeg.p0()-P0File,aSeg.p1()-P0File),
                  ((k==0) ? P8COL::blue : P8COL::black),
                  W()->odisc()
              );
           }
       }
       getchar();
    }
    {
       ELISE_fp aFile(aNameSeg.c_str(),ELISE_fp::WRITE);
       aFile.write(lSeg);
      aFile.close();
    }

}



SarHoughAppli::SarHoughAppli
(
     Tiff_Im      aTif,
     std::string  aName,
     Pt2di        aSzIm
)  :
     mImFile      (aTif.in(0)),
     mNameIm      (aName),
     mSzIm        (aSzIm),
     mSzH         (10,10)
{

}


/*****************************************************************************************/
/*****************************************************************************************/


class SarHoughAppVSar  : public SarHoughAppli
{
      public :
          SarHoughAppVSar
          (
                Tiff_Im  aTif,
                std::string,
                Pt2di sz
          );

          virtual Video_Win * W() {return  &mWinIm;}

      protected :

          Pt2di                    mSzWMax;
          REAL                     mZoomIm;
          Video_Win                mWinIm;
          Video_Display            mDisp;
          virtual Output GrayVisuIm() {return mWinIm.ogray();}

};



SarHoughAppVSar::SarHoughAppVSar
(
     Tiff_Im     aTif ,
     std::string aName,
     Pt2di       aSzIm
)  :
     SarHoughAppli(aTif,aName,aSzIm),
     mSzWMax  (800,700),
     mZoomIm  (mSzWMax.RatioMin(aSzIm)),
     mWinIm(Video_Win::WStd(aSzIm,mZoomIm)),
     mDisp (mWinIm.disp())
{
    mWinIm.set_title("SAR");
}


/*****************************************************************************************/
/*****************************************************************************************/

int main (int argc,char ** argv)
{
/*
   REAL DMin = APPLI_MODE_SAR ? 10.0  : 1.0;
   REAL DMax = APPLI_MODE_SAR ? 200.0 : 400.0;
   REAL VMinRadiom = APPLI_MODE_SAR ? 255.0 : 1.0 ;
*/

   REAL DMin = APPLI_MODE_SAR ? 10.0  : 5.0;
   REAL DMax = APPLI_MODE_SAR ? 200.0 : 400.0;
   REAL VMinRadiom = APPLI_MODE_SAR ? 255.0 : 6.0 ;

   HoughMapedParam aParam 
                   (
                           "/home/pierrot/Data/Ouided/Adapt200.hough",
                           DMin,
                           DMax,
                           VMinRadiom,
                           APPLI_MODE_SAR
                   );

   if (! APPLI_MODE_SAR)
   {
      aParam.mFiltrIncTeta = 0.3;
      aParam.mFiltrEcMaxLoc = 1.0;

      aParam.mVoisRhoMaxLocInit = 1.1;

      aParam.mGradUseFiltreMaxLocDir = true;
      aParam.mGradUseSeuilPreFiltr = false;
      aParam.mUseBCVS = true;
      aParam.mFactCaniche = 0.5;

      aParam.mFiltrSzHoleToFill = 8.0;

   }
/*
   {
      aParam.mFiltrIncTeta = 0.3;
      aParam.mFiltrEcMaxLoc = 1.0;

      aParam.mVoisRhoMaxLocInit = 1.1;

      aParam.mGradUseFiltreMaxLocDir = false;
      aParam.mGradUseSeuilPreFiltr = false;
      aParam.mUseBCVS = false;
      aParam.mFactCaniche = 2.0;

      aParam.mFiltrSzHoleToFill = 1.0;

   }
*/


   std::string  NameHough("/home/data2/pierrot/Hough/Adapt200.hough");
   std::string  NameSar ("../NAV/Chateaudun1.tif"); 
   std::string  NameVis ("/home/pierrot/Data/Ouided/crop.tif");
   std::string  NameFile (APPLI_MODE_SAR ? NameSar : NameVis);
   std::string  NameSegSauv("");

   INT  Visu = 1;


   Pt2di P0   (0,0);
   Pt2di SzIm (-1,-1);

   // REAL       ZoomIm = 2.0;
   INT  VisuAll = 0;

   ElInitArgMain
   (
        argc,argv,
        LArgMain(),
        LArgMain() << EAM(NameHough,"NameHough",true)
                   << EAM(NameFile,"Im",true)
                   << EAM(P0,"P0",true)
                   << EAM(SzIm,"SzIm",true)
                   << EAM(VisuAll,"Visu",true)
                   << EAM(NameSegSauv,"Sauv",true)
   );         



   if (NameSegSauv == "")
      NameSegSauv = StdPrefix(NameFile) +  ".Segs";

   cout << "NameSegSauv = [" << NameSegSauv << "]\n";


   Tiff_Im  aTif(NameFile.c_str());
   if (SzIm == Pt2di(-1,-1))
       SzIm = aTif.Sz2();
   SzIm = Inf(SzIm,aTif.Sz2());

   SarHoughAppli *  anAppli;
   if (Visu)
      anAppli = new  SarHoughAppVSar(aTif,NameFile,SzIm);
   else
      anAppli = new  SarHoughAppli(aTif,NameFile,SzIm);

   anAppli->load(P0,aParam,NameSegSauv);

   getchar();

   // while (1) anAppli.TestHough();


   
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
