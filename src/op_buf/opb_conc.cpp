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


/**********************************************************/
/*                                                        */
/*                    ParamConcOpb                        */
/*                                                        */
/**********************************************************/


INT ParamConcOpb::DefColSmall()
{
    return 1;
}
INT ParamConcOpb::DefColBig()
{
    return 2;
}


bool ParamConcOpb::ToDelete() 
{
    return true;
}
INT ParamConcOpb::ColBig() 
{
    return DefColBig();
}

INT ParamConcOpb::ColSmall(const EliseRle::tContainer&,const Box2di &,INT ColInit) 
{
    return DefColSmall();
}




/**********************************************************/
/*                                                        */
/*                     Conc_OPB_Comp                      */
/*                                                        */
/**********************************************************/



class Conc_OPB_Comp : public Simple_OPBuf1<INT,INT>
{
   typedef INT tIm;
   public :

     Conc_OPB_Comp(ParamConcOpb *,Pt2di SzBox,bool V8);
     virtual ~Conc_OPB_Comp();


   private :

     void make_conc();
     void  calc_buf
           (
               INT **     output,
               tIm ***    input
           );

     virtual Simple_OPBuf1<INT,tIm> * dup_comp();

     Pt2di                 mSzBox;
     bool                  mV8;
     Im2D<tIm,INT>         mIm;
     tIm  **               mData;
     Pt2di                 mSz;
     ParamConcOpb *        mParam;
     INT                   mCbig;
     EliseRle::tContainer  mRles;
     tIm **                mInput;
};


Conc_OPB_Comp::Conc_OPB_Comp(ParamConcOpb * Param,Pt2di SzBox,bool V8) :
   mSzBox (SzBox),
   mV8    (V8),
   mIm    (1,1),
   mData  (0),
   mSz    (1,1),
   mParam (Param),
   mCbig  (mParam->ColBig()),
   mInput (0)
{
   ELISE_ASSERT(mCbig>=0,"Bad color in Conc_OPB_Comp");
}


Conc_OPB_Comp::~Conc_OPB_Comp() 
{
    if (mData==0)
    {
       if (mParam->ToDelete())
          delete mParam;
    }
}



Simple_OPBuf1<INT,INT> * Conc_OPB_Comp::dup_comp()
{
     Conc_OPB_Comp * soc = new Conc_OPB_Comp(mParam,mSzBox,mV8);
     soc->mSz = Pt2di(x1Buf()-x0Buf(),y1Buf()-y0Buf());
     soc->mIm =  Im2D<tIm,INT>(soc->mSz.x,soc->mSz.y);
     soc->mData = soc->mIm.data();

     return soc;
}

void Conc_OPB_Comp::make_conc()
{

   mIm.set_brd(Pt2di(1,1),0);

   for (INT x= 0; x <mSz.x; x++)
   {
        for (INT y= 0; y <mSz.y; y++)
            if (mData[y][x] <0) 
               mData[y][x] =0;
   }


   {
   for (INT x= 0; x <mSz.x; x++)
   {
        for (INT y= 0; y <mSz.y; y++)
        {
            tIm  vXY = mData[y][x] ;
            if (vXY > 0)
            {
                Box2di  box= EliseRle::ConcIfInBox
                             (Pt2di(x,y),mRles,mData,vXY,-mCbig,mV8,mSzBox);
                if (!mRles.empty())
                {
                    INT aCSmall = mParam->ColSmall(mRles,box,vXY);
                    ELISE_ASSERT(aCSmall>=0,"Bad color in Conc_OPB_Comp");
                    if (aCSmall!= mCbig)
                       EliseRle::SetIm(mRles,mData,-aCSmall);
                }
            }
        }
   }
   }

   {
   for (INT x= 0; x <mSz.x; x++)
   {
       for (INT y= 0; y <mSz.y; y++)
       {
           if (mData[y][x] <0) 
              mData[y][x] = - mData[y][x];
           else
              mData[y][x] =  mInput[y+y0Buf()][x+x0Buf()];
       }
   }
   }

}


void Conc_OPB_Comp::calc_buf(INT ** output,tIm *** input)
{
     // ELISE_ASSERT(dim_in()==1,"Multiple Dim in Conc_OPB_Comp::calc_buf");

     if (first_line_in_pack())
     {
        mInput = input[0];

        for (int y = y0Buf() ; y < y1Buf() ; y++)
            convert
            (
                mData[y-y0Buf()],
                mInput[y]+x0Buf(),
                x1Buf()-x0Buf()
            );
        make_conc();
     }

     convert
     (
        output[0]+x0(),
        mData[y_in_pack()-dy0()]-dx0(),
        tx()
     );
}

Fonc_Num  BoxedConc
          (
              Fonc_Num f,
              Pt2di SzBox,
              bool V8,
              ParamConcOpb * param,
              INT nb_pack_y,
              bool aCatInit
          )
{
     return create_op_buf_simple_tpl
            (
               new Conc_OPB_Comp(param,SzBox,V8),
               0,
               f,
               1,
               SzBox+Pt2di(1,1),
               nb_pack_y,
               Simple_OPBuf_Gen::DefOptNPY,
               aCatInit
            );
}


Fonc_Num  BoxedConc(Fonc_Num f,Pt2di SzBox,bool V8,ParamConcOpb * param,bool aCatInit )
{
     SzBox = Pt2di(ElAbs(SzBox.x),ElAbs(SzBox.y));

     INT nb_pack_y = SzBox.y * 3 + 200;
     return BoxedConc(f,SzBox,V8,param,nb_pack_y,aCatInit);
}


Fonc_Num  BoxedConc(Fonc_Num f,Pt2di SzBox,bool V8,bool aCatInit )
{
    return BoxedConc(f,SzBox,V8, new ParamConcOpb,aCatInit);
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
