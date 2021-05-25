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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>
#include "im_tpl/image.h"
#include "im_special/hough.h"

#include "im_tpl/algo_dist32.h"



static std::string TheGlobName = "/media/MYPASSPORT/Documents/Cours/ENSG-TI-IT2/MNT/pleiades/";

double aSeuilBas = 2;
double aSeuilHaut = 5;

class cTestBati : Cont_Vect_Action
{
    public :

        cTestBati(const std::string & aPost,bool Front) :
            mPost (aPost),
            mName (CalcNameResult("")),
            mFile (Tiff_Im::StdConv(mName)),
            mSz   (mFile.sz()),
            mIm   (mSz.x,mSz.y),
            mTIm  (mIm),
            mImCC (mSz.x,mSz.y),
            mImCpt (mSz.x,mSz.y),
            mBWF   (Front ? AllocBW(5) : 0),
            mBWP   (Front ? AllocBW(3) : 0)
        {
             ELISE_COPY ( mFile.all_pts(), mFile.in(), mIm.out() );
             ELISE_COPY ( mIm.border(1), 0, mIm.out() );

        }

        std::string CalcNameResult(const std::string& aStep)
        {
            return  TheGlobName+mPost+ aStep + ".tif";
        }


        void DoAll();
        void TestCC();
        void TestFront();
        void TestDist();
        void TestDist(const Chamfer &,const std::string &);


        void TestSkel(bool inv,int aSurf,double aAng,const std::string &);
        void TestSkel();

        std::string        mPost;
        std::string        mName;
        Tiff_Im            mFile;
        Pt2di              mSz;
        Im2D_U_INT1        mIm;
        TIm2D<U_INT1,INT>  mTIm;
        Im2D_U_INT1        mImCC;
        Im2D_U_INT1        mImCpt;


        Bitm_Win *         mBWF;
        Bitm_Win *         mBWP;

       void action(const ElFifo<Pt2di> &,bool ext); 

        Bitm_Win * AllocBW(int aZ)
        {
              Bitm_Win * aBW = new Bitm_Win("to",RGB_Gray_GlobPal(),mSz*aZ);
              return new Bitm_Win(aBW->chc(Pt2dr(0,0),Pt2dr(aZ,aZ)));
        }

};

void cTestBati::TestCC()
{
    Neighbourhood V8 = Neighbourhood::v8();


    Pt2di aP;
    int aCpt = 1;
    for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
    {
       for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
       {
   
          if (mTIm.get(aP)==255)
          {
             Liste_Pts_INT2 l2(2);


              ELISE_COPY
              (
                  conc
                  (
                     aP,
                     mIm.neigh_test_and_set
                     (
                        V8,
                        255,
                        254,
                        256
                     )
                  ),
                  aCpt,
                  l2 | mImCC.out()
              );

              aCpt++; 
              if (aCpt==256) aCpt=1;
              ELISE_COPY(l2.all_pts(),(l2.card()>100) ? 1 : 2,mImCpt.out());

              if (l2.card()<100)
                  ELISE_COPY(l2.all_pts(),0,mIm.out());
           }
       }
    }

    Tiff_Im::CreateFromIm(mImCC,CalcNameResult("CC_"));
    Tiff_Im::CreateFromIm(mImCpt,CalcNameResult("Cpt_"));

    ELISE_COPY(mIm.all_pts(),255*(mIm.in()!=0),mIm.out());
}

void cTestBati::TestDist(const Chamfer & aC ,const std::string & aStr)
{
      Im2D_U_INT1  aD(mSz.x,mSz.y);
      ELISE_COPY(aD.all_pts(),mIm.in(),aD.out());
      aC.im_dist(aD);

      Tiff_Im::CreateFromIm(aD,CalcNameResult(aStr));
}

void cTestBati::TestDist()
{
     // TestDist(Chamfer::d32,"_Dist32");
     // TestDist(Chamfer::d4,"_Dist4");
     // TestDist(Chamfer::d8,"_Dist8");

      Im2D_INT2  aD(mSz.x,mSz.y);
      ELISE_COPY(aD.all_pts(),mIm.in(),aD.out());
      TIm2D<INT2,INT> aTD(aD);
      aTD.algo_dist_32_neg();
      Tiff_Im::CreateFromIm(aD,CalcNameResult("_Neg32"));
}

void  cTestBati::action(const ElFifo<Pt2di> & aPts,bool ext)
{
     const_cast< ElFifo<Pt2di> &>(aPts).set_circ(true);
     ElFifo<INT>   _approx;
     ArgAPP         _AApp(10.0,40,ArgAPP::D2_droite,ArgAPP::MeanSquare);



    approx_poly(_approx,aPts,_AApp);

    std::cout << "COUCOU "<<  aPts.circ() << " " << aPts.nb() << " " << _approx.nb() << "\n";

    for (int aKP=0 ; aKP<_approx.nb() ; aKP++)
    {
        mBWP->draw_seg
        (
              aPts[_approx[aKP]],
              aPts[_approx[(aKP+1)%_approx.nb()]],
              Line_St(mBWP->prgb()(0,0,255),2.0)
        );
    }
    for (int aKP=0 ; aKP<_approx.nb() ; aKP++)
    {
        mBWP->draw_circle_loc(aPts[_approx[aKP]],1.0,mBWP->prgb()(255,0,0));
    }
}

void cTestBati::TestFront()
{
     if (! mBWF) 
        return; 

     ELISE_COPY(mIm.all_pts(),mIm.in(),mBWF->ogray()|mBWP->ogray());

     //Im2D_U_INT1  aIFr;
      ELISE_COPY
      (
          mIm.all_pts(),
          flag_front8(mIm.in(0)!=0),
          // mBWF->out_graph(mBWF->prgb()(255,0,0),false)
          mBWF->out_graph(Line_St(mBWF->prgb()(255,0,0),2.0),false)
      );

     ELISE_COPY
     (
            mIm.all_pts(),
            cont_vect (mIm.in(),this,true),
            Output::onul()
     );


     mBWF->make_gif(CalcNameResult("_FF").c_str());
     mBWP->make_gif(CalcNameResult("_PP").c_str());

}

void cTestBati::TestSkel(bool inv,int aSurf,double aAng,const std::string & aName)
{
    L_ArgSkeleton aL;
    aL = aL + ArgSkeleton(AngSkel(aAng));
    aL = aL + ArgSkeleton(SurfSkel(aSurf));


    Bitm_Win aBW("toto",RGB_Gray_GlobPal(),mSz);

    ELISE_COPY(mIm.all_pts(),mIm.in(),aBW.ogray());

    ELISE_COPY
    (
        select
        (
             mIm.all_pts(),
             skeleton(inv ? mIm.in(1)==0 : mIm.in(0)!=0, 128,aL)
        ),
        Virgule(Fonc_Num(255),0,0),
        aBW.orgb()
        // Output::onul()
    );
     aBW.make_gif(CalcNameResult(aName).c_str());
}


void  cTestBati::TestSkel()
{
    TestSkel(false,3,0.8,"SkelBarbule");
    TestSkel(false,5,1.7,"SkelStd");
    TestSkel(false,15,2.7,"SkelStrict");

    TestSkel(true,15,2.7,"SkelRoute");
}


void cTestBati::DoAll()
{
     TestCC();
     // TestFront();
     // TestDist();
     TestSkel();
}

void TestImBinaire(const std::string & aName,bool Front)
{
    cTestBati  aTB(aName,Front);
    aTB.DoAll();
}

int main(int argc,char ** argv)
{

     TestImBinaire("Bati",false); 
     //TestImBinaire("Crop_Bati",true); 

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
