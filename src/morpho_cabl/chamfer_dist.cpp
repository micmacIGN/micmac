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



/*******************************************************************/
/******************************************************************/
/*******                                                   ********/
/*******                                                   ********/
/*******          LINES                                    ********/
/*******                                                   ********/
/*******                                                   ********/
/******************************************************************/
/******************************************************************/


/***************************************************************/
/*                                                             */
/*    binarise                                                 */
/*                                                             */
/***************************************************************/

template <class Type> void binarise (Im2D<Type,INT> i2d, INT val)
{
      binarise(i2d.data_lin(),(Type)val,i2d.tx()*i2d.ty());
}

/***************************************************************/
/*                                                             */
/*    chamfer dist for d4,d8,d32                               */
/*                                                             */
/***************************************************************/

template <class Type>  void one_pass_chamfer
                       (
                           Im2D<Type,INT> i2d,
                           const INT   *  vois,
                           INT            nb_v,
                           const INT   *  pds,
                           INT            i0,
                           INT            i1,
                           INT            delta
                       )
{
       Type * d  = i2d.data_lin();

       for (INT i= i0 ; i!= i1 ; i+= delta)
           if (d[i])
           {
              for (INT k=0 ; k<nb_v ; k++)
                  d[i] = ElMin((INT)d[i],d[i+vois[k]]+pds[k]);
           }
}

void  linearise_vois_2d
      (
            INT   *        v_lin,
            const Pt2di *  vois,
            INT            nb_v,
            INT            tx

      )
{

     for (INT i=0 ; i<nb_v ; i++)
         v_lin[i] = vois[i].y * tx + vois[i].x;
}


void Chamfer::im_dist(Im2D<U_INT1,INT> i2d) const
{

     ELISE_COPY(i2d.border(radius()),0,i2d.out());
     binarise (i2d,i2d.vmax()-1);

     INT nb_pts = i2d.tx() * i2d.ty();

     linearise_vois_2d(_v_lin,neigh_yn(),nbv_yn(),i2d.tx());
     one_pass_chamfer(i2d,_v_lin,nbv_yn(),pds_yn(),0,nb_pts,1);

     linearise_vois_2d(_v_lin,neigh_yp(),nbv_yp(),i2d.tx());
     one_pass_chamfer(i2d,_v_lin,nbv_yp(),pds_yp(),nb_pts-1,-1,-1);
}


template <class Type>  void one_pass_etiq
                       (
                           Im2D<Type,INT> i2d,
                           Im2D<INT,INT>  label,
                           const INT   *  vois,
                           INT            nb_v,
                           const INT   *  pds,
                           INT            i0,
                           INT            i1,
                           INT            delta,
                           INT            brd
                       )
{
       Type * d  = i2d.data_lin();
       INT * l   = label.data_lin();

       for (INT i= i0 ; i!= i1 ; i+= delta)
           if (d[i] && d[i] != brd)
           {
              for (INT k=0 ; k<nb_v ; k++)
              {
                  INT nd = d[i+vois[k]]+pds[k];
                  if (nd < d[i])
                  {
                     d[i] = nd;
                     l[i] = l[i+vois[k]];
                  }
              }
           }
}


void Chamfer::dilate_label(Im2D<U_INT1,INT> i2d,Im2D<INT,INT> label,INT vmax) const
{
     Tjs_El_User.ElAssert
     (
          i2d.same_dim_and_sz(label),
          EEM0 << "Incompatible sizes in dilate_label\n"
     );

     ELISE_COPY(label.border(radius()),0,label.out());
     INT nb_pts = i2d.tx() * i2d.ty();

     U_INT1 * d = i2d.data_lin();
     INT *    l = label.data_lin();

     vmax = ElMin(vmax,i2d.vmax()-2);
     for (int xy =0; xy<nb_pts ; xy++)
         d[xy] = l[xy] ? 0 : vmax;

     ELISE_COPY(i2d.border(radius()),vmax+1,i2d.out());


     linearise_vois_2d(_v_lin,neigh_yn(),nbv_yn(),i2d.tx());
     one_pass_etiq(i2d,label,_v_lin,nbv_yn(),pds_yn(),0,nb_pts,1,vmax+1);

     linearise_vois_2d(_v_lin,neigh_yp(),nbv_yp(),i2d.tx());
     one_pass_etiq(i2d,label,_v_lin,nbv_yp(),pds_yp(),nb_pts-1,-1,-1,vmax+1);
}


/***************************************************************/
/*                                                             */
/*    chamfer 32                                               */
/*                                                             */
/***************************************************************/

            // =================================

class Chamfer_32 : public Chamfer
{
      public :
           static const Chamfer_32 TheOne;
      private :

           Chamfer_32() :
             Chamfer(TAB_8_NEIGH,8,p32,VLIN)
           {
           }

        
           static const INT p32[16]; 
           static INT   VLIN[8];

           INT radius () const { return 1;}
};

const INT Chamfer_32::p32[16] = {2,3,2,3,2,3,2,3,
                                 2,3,2,3,2,3,2,3};
const Chamfer_32 Chamfer_32::TheOne;

INT Chamfer_32::VLIN[8];

/***************************************************************/
/*                                                             */
/*    chamfer 8                                                */
/*                                                             */
/***************************************************************/

            // =================================

class Chamfer_8 : public Chamfer
{
      public :
           friend class Chamfer_4; // to grant access to  p1111
           static const Chamfer_8 TheOne;

      private :

           Chamfer_8() :
             Chamfer(TAB_8_NEIGH,8,p1111,VLIN)
           {
           }

        
           static const INT p1111[16]; 
           static INT   VLIN[8];

           INT radius () const { return 1;}
};

const INT Chamfer_8::p1111[16] = {1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1};
const Chamfer_8 Chamfer_8::TheOne;
INT Chamfer_8::VLIN[8];

/***************************************************************/
/*                                                             */
/*    chamfer 4                                                */
/*                                                             */
/***************************************************************/

            // =================================

class Chamfer_4 : public Chamfer
{
      public :
           static const Chamfer_4 TheOne;

      private :

           Chamfer_4() :
             Chamfer(TAB_4_NEIGH,4,Chamfer_8::p1111,VLIN)
           {
           }


           static INT   VLIN[4];
           INT radius () const { return 1;}
};

const Chamfer_4 Chamfer_4::TheOne;
INT Chamfer_4::VLIN[4];


/***************************************************************/
/*                                                             */
/*    chamfer 5711                                             */
/*                                                             */
/***************************************************************/

static Pt2di TAB_5711_NEIGH[32] =
      {

          Pt2di( 1, 0) , Pt2di( 1, 1) , Pt2di( 0, 1) , Pt2di(-1, 1) ,
          Pt2di( 2, 1) , Pt2di( 1, 2) , Pt2di(-1, 2) , Pt2di(-2, 1) ,

          Pt2di(-1, 0) , Pt2di(-1,-1) , Pt2di( 0,-1) , Pt2di( 1,-1) ,
          Pt2di(-2,-1) , Pt2di(-1,-2) , Pt2di( 1,-2) , Pt2di( 2,-1) ,


          Pt2di( 1, 0) , Pt2di( 1, 1) , Pt2di( 0, 1) , Pt2di(-1, 1) ,
          Pt2di( 2, 1) , Pt2di( 1, 2) , Pt2di(-1, 2) , Pt2di(-2, 1) ,

          Pt2di(-1, 0) , Pt2di(-1,-1) , Pt2di( 0,-1) , Pt2di( 1,-1) ,
          Pt2di(-2,-1) , Pt2di(-1,-2) , Pt2di( 1,-2) , Pt2di( 2,-1) 
      };


class Chamfer_5711 : public Chamfer
{
      public :
           static const Chamfer_5711 TheOne;
      private :

           Chamfer_5711() :
             Chamfer(TAB_5711_NEIGH,16,p5711,VLIN)
           {
           }

        
           static const INT p5711[32]; 
           static INT   VLIN[16];

           INT radius () const { return 2;}
};

const Chamfer_5711 Chamfer_5711::TheOne;

const  INT Chamfer_5711::p5711[32] =
       {
               5,  7,  5,  7, 11, 11, 11, 11,
               5,  7,  5,  7, 11, 11, 11, 11,
               5,  7,  5,  7, 11, 11, 11, 11,
               5,  7,  5,  7, 11, 11, 11, 11
       };

INT Chamfer_5711::VLIN[16];

/***************************************************************/
/*                                                             */
/*    chamfer                                                  */
/*                                                             */
/***************************************************************/


const Chamfer & Chamfer::d32 = Chamfer_32::TheOne;
const Chamfer & Chamfer::d8  = Chamfer_8::TheOne;
const Chamfer & Chamfer::d4  = Chamfer_4::TheOne;
const Chamfer & Chamfer::d5711  = Chamfer_5711::TheOne;

const Chamfer & Chamfer::ChamferFromName(const std::string & aName)
{
    if (aName == "4")    return Chamfer_4::TheOne;
    if (aName == "8")    return Chamfer_8::TheOne;
    if (aName == "32")   return Chamfer_32::TheOne;
    if (aName == "5711") return Chamfer_5711::TheOne;

    std::cout << "For name=" << aName << "\n";
    ELISE_ASSERT(false,"Not a valide name for chamfer");
    return Chamfer_4::TheOne;
}

//  p_0_1 must return pds for x =0, y =1;
// this will work with usual chamfer

INT  Chamfer::p_0_1 () const { return  _pds[0];}




Chamfer::Chamfer
(
         const Pt2di * Neigh,
         INT           Nbv,
         const INT *   Pds,
         INT *         v_lin
)   :
        _neigh    (Neigh),
        _nbv      (Nbv),
        _pds      (Pds),
        _neigh_yp (Neigh),
        _nbv_yp   (Nbv/2+1),
        _pds_yp   (Pds),
        _neigh_yn (Neigh+Nbv/2),
        _nbv_yn   (Nbv/2+1),
        _pds_yn   (Pds+Nbv/2),
        _v_lin    (v_lin)
{
}

/***************************************************************/
/*                                                             */
/*                Projection32                                 */
/*                                                             */
/***************************************************************/

cResProj32::cResProj32
(
   Im2D_U_INT2 aDist,
   Im2D_U_INT2 aPX,
   Im2D_U_INT2 aPY,
   bool        aIsInit,
   bool        aIsFull
) :
   mIsInit (aIsInit),
   mIsFull (aIsFull),
   mDist   (aDist),
   mPX     (aPX),
   mPY     (aPY)
{
}

void cResProj32::AssertIsInit() const
{
   ELISE_ASSERT(mIsInit,"cResProj32::AssertIsInit");
}

bool        cResProj32::IsInit() const
{
  return mIsInit;
}

bool        cResProj32::IsFull() const
{
  return mIsFull;
}

Im2D_U_INT2 cResProj32::Dist() const
{
    AssertIsInit();
    return mDist;
}
Im2D_U_INT2 cResProj32::PX() const
{
    AssertIsInit();
    return mPX;
}
Im2D_U_INT2 cResProj32::PY() const
{
    AssertIsInit();
    return mPY;
}


struct cImplemProj32
{
     cImplemProj32
     (
           Im2D_U_INT2 aDist,
           Im2D_U_INT2 aPX,
           Im2D_U_INT2 aPY
     )   :
         mIsInit (false),
         mIsFull (true),
         mTD   (aDist),
         mTPX  (aPX),
         mTPY  (aPY)
     {
     }

     bool              mIsInit;
     bool              mIsFull;
     TIm2D<U_INT2,INT> mTD;
     TIm2D<U_INT2,INT> mTPX;
     TIm2D<U_INT2,INT> mTPY;

     inline void OneStep(Pt2di aP1,Pt2di aP2,int aD)
     {
         if (mTD.get(aP1) > (mTD.get(aP2)+aD))
         {
             mIsInit = true;
             mTD.oset(aP1,mTD.get(aP2)+aD);
             aP1.x --; aP1.y --;
             aP2.x --; aP2.y --;
             mTPX.oset(aP1,mTPX.get(aP2));
             mTPY.oset(aP1,mTPY.get(aP2));
         }
     }

     inline void TestFull(Pt2di aP1)
     {
         if (mTD.get(aP1) )
            mIsFull = false;
     }
};

cResProj32 Projection32(Fonc_Num aF,Pt2di aSz)
{
  Im2D_U_INT2 aDist(aSz.x+2,aSz.y+2);
  Im2D_U_INT2 aPX(aSz.x,aSz.y);
  Im2D_U_INT2 aPY(aSz.x,aSz.y);


  ELISE_COPY
  (
       rectangle(Pt2di(0,0),aSz),
       Virgule(FX,FY),
       Virgule(aPX.out(),aPY.out())
  );
  ELISE_COPY
  (
       rectangle(Pt2di(1,1),aSz+Pt2di(1,1)),
       (trans(aF,Pt2di(-1,-1))==0)*60000,
       aDist.out()
  );
  ELISE_COPY(aDist.border(1),60001,aDist.out());

  cImplemProj32 aIP32(aDist,aPX,aPY);


  Pt2di aP;
  for ( aP.y =1 ; (aP.y<aSz.y+1) && aIP32.mIsFull ; aP.y++)
  {
      for ( aP.x =1 ; aP.x<aSz.x+1 ; aP.x++)
      {
          aIP32.TestFull(aP);
      }
  }
  if (!aIP32.mIsFull)
  {
     for ( aP.y =1 ; aP.y<aSz.y+1 ; aP.y++)
     {
         for ( aP.x =1 ; aP.x<aSz.x+1 ; aP.x++)
         {
             aIP32.OneStep(aP,aP+Pt2di(-1,0),2);  // _d[y][x],(_d[y  ][x-1]-2));
             aIP32.OneStep(aP,aP+Pt2di(0,-1),2);  // _d[y][x],(_d[y-1][x  ]-2));
             aIP32.OneStep(aP,aP+Pt2di(-1,-1),3); // ,(_d[y-1][x-1]-3));
             aIP32.OneStep(aP,aP+Pt2di(1,-1),3); // _d[y][x],(_d[y-1][x+1]-3));
         }
     }

     for ( aP.y =aSz.y ; aP.y>= 1 ; aP.y--)
     {
         for ( aP.x =aSz.x ; aP.x>= 1 ; aP.x--)
         {
             aIP32.OneStep(aP,aP+Pt2di(1,0),2);  // _d[y][x],(_d[y  ][x-1]-2));
             aIP32.OneStep(aP,aP+Pt2di(0,1),2);  // _d[y][x],(_d[y-1][x  ]-2));
             aIP32.OneStep(aP,aP+Pt2di(1,1),3); // ,(_d[y-1][x-1]-3));
             aIP32.OneStep(aP,aP+Pt2di(-1,1),3); // _d[y][x],(_d[y-1][x+1]-3));
         }
     }
  }
  else 
  {
     aIP32.mIsInit = true;
  }

  return cResProj32(aDist,aPX,aPY,aIP32.mIsInit,aIP32.mIsFull);
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
