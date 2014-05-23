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

#include "StdAfx.h"

/*
class cOptimLabelBinaire 
{
    public :

        // Les couts sont entre 0 et 1
        cOptimLabelBinaire(Pt2di aSz,double aDefCost,double aRegul);

        static cOptimLabelBinaire * CoxRoy(Pt2di aSz,double aDefCost,double aRegul);
        static cOptimLabelBinaire * ProgDyn(Pt2di aSz,double aDefCost,double aRegul); // 2 Do


        // 0.0 privilégie l'état 0 ; 1.0 privilégie l'état 1 ....
        void SetCost(Pt2di aP,double aCost);

        virtual Im2D_Bits<1> Sol() = 0;
        virtual ~cOptimLabelBinaire();
        
    protected :
        static U_INT1 ToCost(double aCost);

        Pt2di              mSz;
        Im2D_U_INT1        mCost;  // Memorise les couts entre 0 et 1
        TIm2D<U_INT1,INT>  mTCost;  // Memorise les couts entre 0 et 1
        double             mRegul;
        
        
};
*/


/*******************************************************************/
/*                                                                 */
/*                  cOptimLabelBinaire                             */
/*                                                                 */
/*******************************************************************/

class cCoxRoyOLB : public cOptimLabelBinaire 
{
    public :
         cCoxRoyOLB(Pt2di aSz,double aDefCost,double aRegul);
        Im2D_Bits<1> Sol();
};


cCoxRoyOLB::cCoxRoyOLB(Pt2di aSz,double aDefCost,double aRegul) :
   cOptimLabelBinaire(aSz,aDefCost,aRegul)
{
}



Im2D_Bits<1> cCoxRoyOLB::Sol()
{
    Im2D_Bits<1> aRes(mSz.x,mSz.y);

    Im2D_INT2 aIZMin(mSz.x,mSz.y,0);
    Im2D_INT2 aIZMax(mSz.x,mSz.y,3);

    cInterfaceCoxRoyAlgo * aCox = cInterfaceCoxRoyAlgo::NewOne
                                     (
                                         mSz.x,
                                         mSz.y,
                                         aIZMin.data(),
                                         aIZMax.data(),
                                         true,  // Conx8
                                         false  // UChar
                                     );

     double aMul = 20.0;
     Pt2di aP;
     for (aP.x = 0 ;  aP.x < mSz.x ; aP.x++)
     {
         for (aP.y = 0 ;  aP.y < mSz.y ; aP.y++)
         {
                  aCox->SetCostVert(aP.x,aP.y,0,aMul*0.5);
                  double aCost = 1- mTCost.get(aP) /255.0;
                  aCox->SetCostVert(aP.x,aP.y,1,round_ni(aCost*aMul));
                  aCox->SetCostVert(aP.x,aP.y,2,round_ni(aMul*2));
         }
     }

    aCox->SetStdCostRegul(0,aMul*mRegul,0);

    Im2D_INT2 aISol(mSz.x,mSz.y);
    aCox->TopMaxFlowStd(aISol.data());

    ELISE_COPY(aRes.all_pts(),aISol.in()!=0,aRes.out());


    delete aCox;

    return aRes;
}
   



/*******************************************************************/
/*                                                                 */
/*                  cOptimLabelBinaire                             */
/*                                                                 */
/*******************************************************************/

cOptimLabelBinaire::cOptimLabelBinaire(Pt2di aSz,double aDefCost,double aRegul) :
      mSz    (aSz),
      mCost  (aSz.x,aSz.y,ToCost(aDefCost)) ,
      mTCost (mCost),
      mRegul (aRegul)
{
}

U_INT1 cOptimLabelBinaire::ToCost(double aCost)
{
    return ElMax(0,ElMin(255,round_ni(255*aCost)));
}

void cOptimLabelBinaire::SetCost(Pt2di aP,double aCost)
{
    mTCost.oset(aP,aCost);
}

cOptimLabelBinaire::~cOptimLabelBinaire()
{
}

cOptimLabelBinaire * cOptimLabelBinaire::CoxRoy(Pt2di aSz,double aDefCost,double aRegul)
{
    return new cCoxRoyOLB(aSz,aDefCost,aRegul);
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
