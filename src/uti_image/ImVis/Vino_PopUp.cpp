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

#include "Vino.h"


#if (ELISE_X11)

void cAppli_Vino::End()
{
    std::cout << "   ******************************************\n";
    std::cout << "   *                                        *\n";
    std::cout << "   *    V-isualizer of                      *\n";
    std::cout << "   *    I-mages                             *\n";
    std::cout << "   *    N-ot                                *\n";
    std::cout << "   *    O-versized                          *\n";
    std::cout << "   *                                        *\n";
    std::cout << "   ******************************************\n";

    exit(EXIT_SUCCESS);

}

void cAppli_Vino::HistoSetDyn()
{
    mW->fill_rect(Pt2dr(5,20),Pt2dr(280,60),mW->pdisc()(P8COL::magenta));
    mW->fixed_string(Pt2dr(40,45),"Clik P1 and P2 of rectangle",mW->pdisc()(P8COL::black),true);

    Pt2dr aP1 = mW->clik_in()._pt;
    mW->fill_rect(aP1-Pt2dr(3,3),aP1+Pt2dr(3,3),mW->pdisc()(P8COL::green));
    Pt2dr aP2 = mW->clik_in()._pt;
    mW->draw_rect(aP1,aP2,mW->pdisc()(P8COL::green));
}



void  cAppli_Vino::MenuPopUp()
{
    mPopUpCur = 0;
    if ((!mCtrl0) && (!mShift0)) mPopUpCur = mPopUpBase;

    if (mPopUpCur==0)  return;

    mModeGrab=eModeVinoPopUp;

    mPopUpCur->UpCenter(Pt2di(mP0Click));
    mW->grab(*this);
    mCaseCur = mPopUpCur->PopAndGet();


    if (mPopUpCur==mPopUpBase)
    {
        if (mCaseCur== mCaseExit)
        {
            End();
        }

        if (    (mCaseCur==mCaseHStat)
             || (mCaseCur==mCaseHMinMax)
             || (mCaseCur==mCaseHEqual)
           )
        {
            HistoSetDyn();
        }
    }
}


Im2D_Bits<1> cAppli_Vino::Icone(const std::string & aName,const Pt2di & aSz)
{
   cElBitmFont & aFont = cElBitmFont::BasicFont_10x8() ;

   return aFont.MultiLineImageString(aName,Pt2di(0,5),-aSz,0);
}

CaseGPUMT * cAppli_Vino::CaseBase(const std::string& aName,const Pt2di aNumCase)
{
   Im2D_Bits<1> anIc = Icone(aName,mSzCase);
   ELISE_COPY(anIc.border(3),1,anIc.out());
   ELISE_COPY(anIc.border(1),0,anIc.out());

/*
   Pt2di  aSz = anIc.sz();
   Im2D_U_INT1 aRes(aSz.x,aSz.y);
   ELISE_COPY
   (
         aRes.all_pts(),
         Max(anIc.in(),rect_som(anIc.in(0),1)/9.0) *255,
         aRes.out()
   );

   return new CaseGPUMT (*mPopUpBase,"i",aNumCase, 255-aRes.in(255));
*/
    return new CaseGPUMT (*mPopUpBase,"i",aNumCase, (!anIc.in(0)) *255);
}



void cAppli_Vino::InitMenu()
{
    mSzCase        = Pt2di(70,40);
    mPopUpBase = new GridPopUpMenuTransp(*mW,mSzCase,Pt2di(5,3),Pt2di(1,1));

    mCaseExit  = CaseBase("Exit",Pt2di(0,0));
    mCaseHStat  = CaseBase("Histo\nStat2",Pt2di(1,0));
    mCaseHMinMax  = CaseBase("Histo\nMinMax",Pt2di(2,0));
    mCaseHEqual   = CaseBase("Histo\nEqual",Pt2di(3,0));

}






#endif



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
