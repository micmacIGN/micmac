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



Liste_Pts_U_INT2 Skeleton
(
     U_INT1 **    result,
     U_INT1 **    image,
     int                 tx,
     int                 ty,
     int                 surf_threshlod,
     double              angular_threshlod,
     bool                skel_of_disk,
     bool                prolgt_extre,
     bool                with_result,
     bool                cx8,
     U_INT2 **           tmp
)
{
      ResultVeinSkel res =
      VeinerizationSkeleton
      (
               result,
               image,
               tx,
               ty,
               surf_threshlod,
               angular_threshlod,
               skel_of_disk,
               prolgt_extre,
               with_result,
               cx8,
               tmp
      );

      if (with_result)
      {
          Liste_Pts_U_INT2 l(res.x,res.y,res.nb);
          freeResultVeinSkel(&res);
          return l;
      }
      else
      {
          freeResultVeinSkel(&res);
          return Liste_Pts_U_INT2(2);
      }
}

//=====================================================================




class Modif_Arg_Skel : public RC_Object
{
      public :

          typedef enum mode
          {
               ang,
               surf,
               skdisk,
               prolgt,
               result,
               cx8,
               tmp
          } mode;

          void modifASk(Data_ArgSkeleton &);

          Modif_Arg_Skel(REAL r,mode theM) : _r (r) , _theM (theM) {}
          Modif_Arg_Skel(INT  i,mode theM) : _i (i) , _theM (theM) {}
          Modif_Arg_Skel(bool  b,mode theM) : _b (b) , _theM (theM) {}
          Modif_Arg_Skel(Im2D_U_INT2  i);


          REAL        _r;
          INT         _i;
          bool        _b;
          U_INT2 ** _ui2;

          mode _theM;
          
};

Modif_Arg_Skel::Modif_Arg_Skel(Im2D_U_INT2  i) :
     _ui2  (i.data()),
     _theM (tmp)
{
}

void Modif_Arg_Skel::modifASk(Data_ArgSkeleton & DAS)
{
     switch(_theM)
     {
          case ang :
               DAS._ang  = _r;
          break;

          case surf :
               DAS._surf  = _i;
          break;

          case skdisk :
               DAS._skel_of_disk  = _b;
          break;

          case prolgt :
               DAS._prolgt_extre  = _b;
          break;

          case result :
               DAS._result  = _b;
          break;

          case cx8 :
               DAS._cx8  = _b;
          break;

          case tmp :
               if (!  DAS._tmp)
                  DAS._tmp  = _ui2;
          break;
     }
}

Data_ArgSkeleton::Data_ArgSkeleton(INT tx,INT ty,L_ArgSkeleton larg) :
  _tx              (tx),
  _ty              (ty),
  _ang             (2.1),
  _surf            (6),
  _skel_of_disk    (false),
  _prolgt_extre    (false),
  _result          (false),
  _cx8             (true),
  _tmp             (0)
{
      for (; !larg.empty() ; larg = larg.cdr())
          larg.car().das()->modifASk(*this);
}

               //=========================

ArgSkeleton::ArgSkeleton( Modif_Arg_Skel * ptr) :
      PRC0(ptr)
{
}

AngSkel::AngSkel(REAL ang) :
       ArgSkeleton(new Modif_Arg_Skel(ang,Modif_Arg_Skel::ang))
{
}

SurfSkel::SurfSkel(int surf) :
       ArgSkeleton(new Modif_Arg_Skel(surf,Modif_Arg_Skel::surf))
{
}


SkelOfDisk::SkelOfDisk(bool skd) :
       ArgSkeleton(new Modif_Arg_Skel(skd,Modif_Arg_Skel::skdisk ))
{
}

ProlgtSkel::ProlgtSkel(bool prol) :
       ArgSkeleton(new Modif_Arg_Skel(prol,Modif_Arg_Skel::prolgt ))
{
}

Cx8Skel::Cx8Skel(bool cx8) :
       ArgSkeleton(new Modif_Arg_Skel(cx8,Modif_Arg_Skel::cx8 ))
{
}

ResultSkel::ResultSkel(bool result) :
       ArgSkeleton(new Modif_Arg_Skel(result,Modif_Arg_Skel::result ))
{
}

TmpSkel::TmpSkel(Im2D_U_INT2 b) :
       ArgSkeleton(new Modif_Arg_Skel(b))
{
}

//=====================================================================

Liste_Pts_U_INT2 Skeleton
(
     U_INT1      ** skel,
     U_INT1      ** image,
     INT            tx,
     INT            ty,
     L_ArgSkeleton  larg
)
{

     Data_ArgSkeleton  arg(tx,ty,larg);

       return
            Skeleton
            (
                skel,
                image,
                tx,
                ty,
                arg._surf,
                arg._ang,
                arg._skel_of_disk,
                arg._prolgt_extre,
                arg._result,
                arg._cx8,
                arg._tmp
            );
}

const Pt2di ElSzDefSkel (-1,-1);

Liste_Pts_U_INT2 Skeleton
(
     Im2D_U_INT1  skel,
     Im2D_U_INT1  image,
     L_ArgSkeleton  larg,
     Pt2di          SZ
)
{
     if (SZ == ElSzDefSkel)
        SZ = image.sz();
     SZ = Inf(SZ,Inf(skel.sz(),image.sz()));

     return Skeleton(skel.data(),image.data(),SZ.x,SZ.y,larg);
}



Im1D_U_INT1 NbBits(INT nbb)
{
    INT nbv = (1<<nbb);
    Im1D_U_INT1 res (nbv,0);
    U_INT1 * d = res.data();

    for(int v=0; v<nbv ; v++)
       for (int b=0; b<nbb ; b++)
           if (v & (1<<b))
              d[v]++;

    return res;
}


const ElList<ArgSkeleton> ArgSkeleton::L_empty;

L_ArgSkeleton NewLArgSkel(ArgSkeleton anArg)
{
   return L_ArgSkeleton() + anArg;
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
