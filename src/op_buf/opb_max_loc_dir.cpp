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
/*                     MaxLocDir_OPB                      */
/*                                                        */
/**********************************************************/




class MaxLocDir_OPB : public Simple_OPBuf1<REAL,REAL>
{
   public :

     MaxLocDir_OPB(REAL OuvAng,Pt2di P0,Pt2di p1,bool OrientedMaxLoc,REAL RhoCalc);
     virtual ~MaxLocDir_OPB(){}


   private :
     enum {NbTeta = 256,NbBits = 8};

     INT Teta2I(REAL Teta){return mod(round_ni((Teta*NbTeta)/(2*PI)),NbTeta);}
     REAL I2Teta(INT iTeta){return (iTeta*2*PI)/NbTeta;}
     bool IsMaxLoc(INT x,INT iteta);
     INT  DirComplem(INT iteta) {return mod(iteta+NbTeta/2,NbTeta);}

     Simple_OPBuf1<REAL,REAL> * dup_comp();

     void make_Vois();
     void AddVois(INT);
     void  calc_buf (REAL **  output,REAL ***  input);

  
     TIm2D<REAL8,REAL8>    mIm;
     ElSTDNS vector<ElPFixed<NbBits> > mVois;
     ElSTDNS vector<INT>               mIndVois;
     REAL8                     mROuvAng;
     INT                       mIOuvAng;
     bool                      mOrientedMaxLoc;
     REAL8                     mRhoCalc;

};

MaxLocDir_OPB::MaxLocDir_OPB(REAL OuvAng,Pt2di P0,Pt2di P1,bool OrientedMaxLoc,REAL RhoCalc) :
     mIm             (0,P0,P1),
     mROuvAng        (OuvAng),
     mIOuvAng        (Teta2I(OuvAng)),
     mOrientedMaxLoc (OrientedMaxLoc),
     mRhoCalc        (ElMin(0.99,ElAbs(RhoCalc)))
{
}


Simple_OPBuf1<REAL,REAL> * MaxLocDir_OPB::dup_comp()
{
    MaxLocDir_OPB * res = new MaxLocDir_OPB
                              (
                                  mROuvAng,
                                  Pt2di(x0Buf(),y0Buf()),
                                  Pt2di(x1Buf(),y1Buf()),
                                  mOrientedMaxLoc,
                                  mRhoCalc
                              );

   res->make_Vois();

   return res;
}



bool MaxLocDir_OPB::IsMaxLoc(INT x,INT iteta)
{
    Pt2di p0(x,0);
    REAL v0b2 = mIm.get(p0)* (1<<(2*NbBits));

    INT kDeb=mIndVois[iteta];
    INT kFin=mIndVois[iteta+1];

    for (INT k=kDeb;k<kFin ; k++)
    {
       REAL vb2 = TImGet<REAL8,REAL,NbBits>::getb2(mIm,mVois[k]+p0);
       if (vb2 >= v0b2)
          return false;
    }
    return true;
}

void MaxLocDir_OPB::AddVois(INT Iteta)
{
    mVois.push_back
    (
       ElPFixed<NbBits>
       (
          Pt2dr::FromPolar
          (
             mRhoCalc,
             I2Teta(Iteta)
          )
       )
    );
}



void MaxLocDir_OPB::make_Vois()
{
    mIndVois.push_back(0);
    INT Step8Vois = NbTeta/8;
    for (INT iTeta=0; iTeta<NbTeta ; iTeta++)
    {
         INT iTeta1 = iTeta-mIOuvAng;
         INT iTeta2 = iTeta+mIOuvAng;
         for (INT iT8v=0; iT8v<iTeta2; iT8v+=Step8Vois)
             if (iT8v>iTeta1)
                AddVois(iT8v);
         AddVois(iTeta1);
         AddVois(iTeta2);
         mIndVois.push_back((int) mVois.size());
    }
}


void  MaxLocDir_OPB::calc_buf (REAL **  output,REAL ***  input)
{
     REAL ** Mod = input[0];
     REAL *  Ang = input[1][0];
     REAL *  res = output[0];

     mIm.SetData(Mod);

     for (INT x=x0(); x<x1() ; x++)
     {
         INT iTeta = Teta2I(Ang[x]);
         res[x] =      IsMaxLoc(x,iTeta) 
                   &&  (mOrientedMaxLoc || IsMaxLoc(x,DirComplem(iTeta)));
     }
}


Fonc_Num  GenMaxLocDir
          (
              Fonc_Num f,
              bool OuvAngCste,
              REAL OuvAng,
              bool OrientedMaxLoc,
              REAL RhoCalc,
              bool aCatInit,
              bool ConvInt
          )
{
     ELISE_ASSERT
     (
         f. dimf_out() ==(OuvAngCste ? 2: 3),
         "Bad Dim in MaxLocDir"
     );

     Fonc_Num res =
               create_op_buf_simple_tpl
               (
                  0,
                  new MaxLocDir_OPB(OuvAng,Pt2di(0,0),Pt2di(0,0),OrientedMaxLoc,RhoCalc),
                  f,
                  1,
                  Pt2di(1,1),
                  Simple_OPBuf_Gen::DefNbPackY,
                  Simple_OPBuf_Gen::DefOptNPY,
                  aCatInit
               );

     return ConvInt ? Iconv(res) : res;
}



Fonc_Num  MaxLocDir(Fonc_Num f,REAL OuvAng,bool OrientedMaxLoc,REAL RhoCalc, bool aCatInit )
{
    return GenMaxLocDir(f,true,OuvAng,OrientedMaxLoc,RhoCalc,aCatInit,true);
}


Fonc_Num  RMaxLocDir(Fonc_Num f,REAL OuvAng,bool OrientedMaxLoc,REAL RhoCalc, bool aCatInit )
{
    return GenMaxLocDir(f,true,OuvAng,OrientedMaxLoc,RhoCalc,aCatInit,false);
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
