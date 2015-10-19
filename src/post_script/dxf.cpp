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



      //===================================================
      //===================================================
      //===================================================
      //===================================================




class DATA_DXF_WRITER :  public RC_Object
{
      public :

          DATA_DXF_WRITER
          (
               const ElSTDNS string &,
               Box2di        ,
               bool         InvY = true
          );

          virtual ~DATA_DXF_WRITER();


          void PutPt0(Pt2dr);
          void PutSeg(Seg2d s,const char * Layer);
          void PutVertex(Pt2dr,const char * Layer);
          void PutPolyline(const ElFilo<Pt2dr> &,const char * Layer,bool circ = false);

      private :

          void PutPt(Pt2dr,INT DTag,bool corY = true);
          void PutPt1(Pt2dr);
          void PutPt0(Pt2dr,bool corY);

          ElSTDNS string  _name;
          FILE *  _fp;
          bool    _inv_Y;
          REAL    _Y1;

          REAL StdY(REAL y) {return _inv_Y ? (_Y1 -y) : y;}

          Pt2dr StdPt(Pt2dr p){return Pt2dr(p.x,StdY(p.y));}


          typedef enum  
          {
               TagStrEntity =       0,
               TagStrBlock =        2,
               TagStrLayer =        8,
               TagStrIdent =        9,
               TagPtX0 =            10,
               TagPtY0 =            20,
               TagEntityFollow =    66,
               TagCount =           70
          } TAG;

          static const char *_SECTION;
          static const char *_ENDSEC;
          static const char *_SEQEND;
          static const char *_EOF;
          static const char *_HEADER;
          static const char *_BLOCKS;
          static const char *_TABLES;
          static const char *_EXTMIN;
          static const char *_EXTMAX;
          static const char *_LUPREC;
          static const char *_ENTITIES;
          static const char *_LINE;
          static const char *_POLYLINE;
          static const char *_VERTEX;


      //==========================================


          void PutTag(TAG tag) { fprintf(_fp,"%3d\n",tag);}
          void PutReal(REAL r) {fprintf(_fp,"%.3f\n",r); }
          void PutInt(INT  i) {fprintf(_fp,"  %d\n",i); }
          void PutValY(REAL y) {PutReal(StdY(y));}
          void PutStr(const char * str) {fprintf(_fp,"%s\n",str);}



          void PutTagX0()     {PutTag(TagPtX0);}
          void PutTagY0()     {PutTag(TagPtY0);}
          void PutTagLayer()  {PutTag(TagStrLayer);}
          void PutTagEntity() {PutTag(TagStrEntity);}
          void PutTagBlock()  {PutTag(TagStrBlock);}
          void PutTagIdent()  {PutTag(TagStrIdent);}

          void PutCount(INT i)
          {
               PutTag(TagCount);
               PutInt(i);
          }

          void EmptySec(const char * str)
          {
               PutSection();
               PutStrBlock(str),
               PutEndSec();
          }


          void PutLayer(const char * str)
          {
               if (str)
               {
                    PutTagLayer();
                    PutStr(str);
               }
          }


          void PutStrEntity(const char * str)
          {
                PutTagEntity();
                PutStr(str);
          }
          void PutSection()     {PutStrEntity(_SECTION);}
          void PutEndSec()      {PutStrEntity(_ENDSEC);}
          void PutEOF()         {PutStrEntity(_EOF);}
          void PutSeqEnd() {PutStrEntity(_SEQEND);}

          void PutStrBlock(const char * str)
          {
                PutTagBlock();
                PutStr(str);
          }
          void PutStrIdent(const char * str)
          {
                PutTagIdent();
                PutStr(str);
          }

};


const char * DATA_DXF_WRITER::_SEQEND       = "SEQEND"   ;
const char * DATA_DXF_WRITER::_SECTION      = "SECTION"  ;
const char * DATA_DXF_WRITER::_ENDSEC       = "ENDSEC"   ;
const char * DATA_DXF_WRITER::_HEADER       = "HEADER"   ;
const char * DATA_DXF_WRITER::_BLOCKS       = "BLOCKS"   ;
const char * DATA_DXF_WRITER::_TABLES       = "TABLES"   ;
const char * DATA_DXF_WRITER::_EXTMIN       = "$EXTMIN"  ;
const char * DATA_DXF_WRITER::_EXTMAX       = "$EXTMAX"  ;
const char * DATA_DXF_WRITER::_LUPREC       = "$LUPREC"  ;
const char * DATA_DXF_WRITER::_ENTITIES     = "ENTITIES" ;
const char * DATA_DXF_WRITER::_LINE         = "LINE"     ;
const char * DATA_DXF_WRITER::_POLYLINE     = "POLYLINE" ;
const char * DATA_DXF_WRITER::_EOF          = "EOF"      ;
const char * DATA_DXF_WRITER::_VERTEX       = "VERTEX"   ;


void  DATA_DXF_WRITER::PutPt(Pt2dr p,INT dtag,bool corY)
{
      PutTag((TAG)(TagPtX0+dtag));
      PutReal(p.x);
      PutTag((TAG)(TagPtY0+dtag));
      if (corY)
         PutValY(p.y);
      else
        PutReal(p.x);
}

void  DATA_DXF_WRITER::PutPt0(Pt2dr p)
{
     PutPt(p,0);
}

void  DATA_DXF_WRITER::PutPt0(Pt2dr p,bool corY)
{
     PutPt(p,0,corY);
}


void  DATA_DXF_WRITER::PutPt1(Pt2dr p)
{
     PutPt(p,1);
}



DATA_DXF_WRITER::DATA_DXF_WRITER
(
     const ElSTDNS string & Name,
     Box2di         Box,
     bool           InvY 
)   :
    _name  (Name),
    _fp    (ElFopen(Name.c_str(),"wb")),
    _inv_Y (InvY),
    _Y1    (Box._p1.y)
{
    Box2di B(   StdPt(Pt2dr(Box._p0))    ,   StdPt(Pt2dr(Box._p1))    );
    ELISE_ASSERT(_fp != 0,"Can't open DXF file");

    PutSection();
    PutStrBlock(_HEADER);
    PutStrIdent(_EXTMIN);
    PutPt0(Pt2dr(B._p0),false);
    PutStrIdent(_EXTMAX);
    PutPt0(Pt2dr(B._p1),false);
    PutStrIdent(_LUPREC);
    PutCount(14);
    PutEndSec();

    EmptySec(_TABLES); 
    EmptySec(_BLOCKS); 

    PutSection();
    PutStrBlock(_ENTITIES);
}


void DATA_DXF_WRITER::PutSeg(Seg2d seg,const char * Layer)
{
    PutStrEntity(_LINE);
    PutLayer(Layer);
    PutPt0(seg.p0());
    PutPt1(seg.p1());
}

void DATA_DXF_WRITER::PutVertex(Pt2dr pt,const char * Layer)
{
    PutStrEntity(_VERTEX);
    PutLayer(Layer);
    PutPt0(pt);
}


DATA_DXF_WRITER::~DATA_DXF_WRITER()
{
      PutEndSec();
      PutEOF();
      ElFclose(_fp);
}

void DATA_DXF_WRITER::PutPolyline(const ElFilo<Pt2dr> & pts,const char * Layer,bool circ)
{
    PutStrEntity(_POLYLINE);
    PutLayer(Layer);
    PutTag(TagEntityFollow);
    PutInt(1);

    INT nb = pts.nb() + (circ  ? 1 : 0);

     for (INT k=0; k<nb ; k++)
         PutVertex(pts[k%pts.nb()],Layer);
     PutSeqEnd();
}

      //===================================================
      //===================================================
      //===================================================
      //===================================================

DXF_Writer::DXF_Writer
(
     const char *     str,
     Box2di           box,
     bool             InvY 
)  :
   PRC0(new DATA_DXF_WRITER(str,box,InvY))
{
}

DATA_DXF_WRITER  * DXF_Writer::ddw()
{
    return (DATA_DXF_WRITER  *) _ptr;
}
void  DXF_Writer::PutPt0(Pt2dr pt)
{
     ddw()->PutPt0(pt);
}
void  DXF_Writer::PutSeg(Seg2d s,const char * Layer)
{
      ddw()->PutSeg(s,Layer);
}
void  DXF_Writer::PutVertex(Pt2dr p,const char * Layer)
{
      ddw()->PutVertex(p,Layer);
}
void   DXF_Writer::PutPolyline
       (
           const ElFilo<Pt2dr> & pts,
           const char * Layer,
           bool circ
       )
{
      ddw()->PutPolyline(pts,Layer,circ);
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
