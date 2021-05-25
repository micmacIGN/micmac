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

#include "XML_GEN/all.h"
using namespace NS_ParamChantierPhotogram;




class cAppliSM : public Grab_Untill_Realeased
{
    public :
         cAppliSM(int argc,char ** argv);

	 void NoOp(){}
	 void ExeClik(Clik aCl);
    private :
        void  GUR_query_pointer(Clik,bool);
        void ShowCC();

        void GetPolyg(Clik aCl);
        void DoMenu(Clik aCl);

        Pt2di                  mSzWP;
        std::string            mDir;
        std::string            mNIm;
        int                    mCurCoul;

	Tiff_Im *           mT1;
	Video_Win*          mW;
	Video_Win*          mWCC;
        EliseStdImageInteractor * mESII;
        BiScroller *              mBiS;

        Tiff_Im                 mTifExit;
        Pt2di                   mSzCase;
        
        GridPopUpMenuTransp*    mPopUp;
        CaseGPUMT *             mCaseExit;
        BoolCaseGPUMT *         mCaseCoul;
        bool                    mEnd;
        std::string             mAttr;
    
};

void cAppliSM::GUR_query_pointer(Clik cl,bool)
{
    mPopUp->SetPtActif(Pt2di(cl._pt));
}

void cAppliSM::GetPolyg(Clik aCl)
{
   std::vector<Pt2dr> aVPt =
        mESII->GetPolyg(P8COL::red,P8COL::green,mESII->W2U(aCl._pt));
    if (int(aVPt.size()) > 2)
    {
       ElList<Pt2di> aLPt;

       for (int aK=0 ; aK<int(aVPt.size()) ; aK++)
           aLPt = aLPt + round_ni(aVPt[aK]);
       ELISE_COPY
       (
         polygone(aLPt),
         mCurCoul,
         mBiS->ImMasq().out()
       );
    }
    mESII->Refresh();
}


void cAppliSM::ShowCC()
{
    Box2dr aBox(Pt2di(0,0),mWCC->sz());

    Fill_St aFst(mWCC->pdisc()(mCurCoul? P8COL::yellow:P8COL::white));
    mWCC->fill_rect(aBox._p0,aBox._p1,aFst);

    mWCC->draw_rect(aBox,Line_St(mWCC->pdisc()(P8COL::red),4));
}

void cAppliSM::DoMenu(Clik aCl)
{
   mPopUp->UpCenter(Pt2di(aCl._pt));
   mW->grab(*this);
   CaseGPUMT * aCase =  mPopUp->PopAndGet();

   if (aCase== mCaseExit)
   {
      mEnd=true;
   }
   else if (aCase==mCaseCoul)
   {
       mCurCoul = int(mCaseCoul->Val());
       ShowCC();
   }
   else
   {
   }
}

void cAppliSM::ExeClik(Clik aCl)
{
    std::cout  << mESII->W2U(aCl._pt)  << aCl._b << "\n";
    if (aCl._b==1)
    {
       GetPolyg(aCl);
    }
    else if (aCl._b==3)
    {
       DoMenu(aCl);
    }
}

cAppliSM::cAppliSM(int argc,char ** argv) :
   mCurCoul  (1),
   // mTifExit  ("data/Exit.tif"),
   mTifExit  (MMIcone("Exit")),
   mSzCase   (mTifExit.sz()),
   mEnd      (false)
{
    mSzWP   = Pt2di(900,700);
    std::string aFullName;
    std::string aPost("Masq");
    std::string aNameMasq ="";
    double aGama=1.0;
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAM(aFullName) ,
           LArgMain() << EAM(mSzWP,"SzW",true)
                      << EAM(aPost,"Post",true)
                      << EAM(aNameMasq,"Name",true)
                      << EAM(aGama,"Gama",true)
                      << EAM(mAttr,"Attr",true)
    );

    aPost = aPost + mAttr;

    SplitDirAndFile(mDir,mNIm,aFullName);
    std::string aTifPost;
    if (IsPostfixed(mNIm))
    {
       aTifPost = StdPostfix(mNIm);
    }
    if (! IsKnownTifPost(aTifPost))
    {
      aTifPost = "tif";
    }

    aTifPost = "." + aTifPost;

    // std::string aNTifIm = mDir+mNIm;
    std::string aNTifIm = NameFileStd(mDir+mNIm,1,false);

    if (aNameMasq=="")
       aNameMasq = mDir+StdPrefixGen(mNIm)+ "_" + aPost + aTifPost;
    else 
    {
       // aNameMasq = mDir+aNameMasq ;
    }

    if (!ELISE_fp::exist_file(aNameMasq))
    {
         Tiff_Im::SetDefTileFile(1<<20);
         Tiff_Im aFIm=Tiff_Im::StdConv(aNTifIm);
         Tiff_Im aFM
                 (   aNameMasq.c_str(),
                      aFIm.sz(),
                      GenIm::u_int1,
                      Tiff_Im::No_Compr,
                      Tiff_Im::BlackIsZero
                 );

         ELISE_COPY(aFM.all_pts(),0,aFM.out());
    }

    mT1  = new Tiff_Im(Tiff_Im::StdConvGen(aNTifIm,1,false));


    {
        std::string aNameXML = StdPrefix(aNameMasq)+".xml";
        if (!ELISE_fp::exist_file(aNameXML))
        {
           cFileOriMnt anOri;

           anOri.NameFileMnt() = aNameMasq;
           anOri.NombrePixels() = mT1->sz();
           anOri.OriginePlani() = Pt2dr(0,0);
           anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
           anOri.OrigineAlti() = 0.0;
           anOri.ResolutionAlti() = 1.0;
           anOri.Geometrie() = eGeomMNTFaisceauIm1PrCh_Px1D;

           MakeFileXML(anOri,aNameXML);
        }
        std::cout << aNameXML << "\n";
    }



    mW =  Video_Win::PtrWStd(mSzWP);
    mWCC = new Video_Win(Video_Win(*mW,Video_Win::eDroiteH,Pt2di(100,50)));

    VideoWin_Visu_ElImScr * aVVE = new VideoWin_Visu_ElImScr(*mW,*mT1);
    // ElPyramScroller * aPyr = ElImScroller::StdPyramide(*aVVE,aNTifIm);

// std::cout << "EEEEEEEEEEEE  "<< aNTifIm << "\n";
    //mBiS =  BiScroller::MasqBiscroller(*aVVE,aNTifIm,aNameMasq);


    std::vector<Elise_colour> aVC;
    aVC.push_back(Elise_colour::white);  // 0 : unused
    aVC.push_back(Elise_colour::green);
    aVC.push_back(Elise_colour::red);
    aVC.push_back(Elise_colour::orange);

    mBiS  = BiScroller::LutColoredBiscroller(*aVVE,aNTifIm,aNameMasq,&(aVC[0]),aVC.size());

   mBiS->GraySetGamaCorrec(aGama);

    mESII = new EliseStdImageInteractor(*mW,*mBiS,2,5,4);


    mPopUp = new GridPopUpMenuTransp(*mW,mSzCase,Pt2di(5,5),Pt2di(1,1));

    
    mCaseExit = new CaseGPUMT
                    (
                       *mPopUp,"titi",Pt2di(4,4),
                       MMIcone("Exit").in(0) *255
                    );

    mCaseCoul  = new BoolCaseGPUMT
                    (
                       *mPopUp,"titi",Pt2di(0,0),
                       MMIcone("Coul").in(0) *255,
                       (!MMIcone("Coul").in(0)) *255,
                       mCurCoul != 0
                    );

    ShowCC();

    while (! mEnd)
    {
         ExeClik(mESII->clik_press())  ;
    }


    Tiff_Im aTifM = Tiff_Im::StdConv(aNameMasq);

//   std::cout  << "MM " << aMin << " " << aMax << "\n";
    ELISE_COPY
    (
       mBiS-> ImMasq().all_pts(),
       mBiS-> ImMasq().in() ,
       aTifM.out()
    );

}


int main(int argc,char ** argv)
{
    cAppliSM aAP(argc,argv);
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
