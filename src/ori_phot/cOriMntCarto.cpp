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

/*********************************************/
/*                                           */
/*             cOriMntCarto                  */
/*                                           */
/*********************************************/
static std::string  LireTexte(FILE * aFp,const char *  Expected)
{
    char Buf[300];
    VoidFscanf ( aFp, "%s", Buf);

    std::string aRes(Buf);
    if (Expected)
       ELISE_ASSERT
       (
             aRes==std::string(Expected),
             "Unexpected string in LireTexte"
       );

    return aRes;
}
static long long int LireInt(FILE * aFp)
{
    long long int aVal;
    #if (ELISE_MinGW)
        VoidFscanf ( aFp, "%I64d",&aVal);
    #else
        VoidFscanf ( aFp, "%lld",&aVal);
    #endif
    return aVal;
}

static double LireIntAsReal(FILE * aFp,double aResol=1.0)
{
    return LireInt(aFp) / aResol;
}


const REAL cOriMntCarto::UniteFile = 1e-3;

double  cOriMntCarto:: StdLireIntAsReal(FILE * aFp)
{
   return  LireIntAsReal(aFp,1.0/UniteFile);
}

REAL cOriMntCarto::ToUniteOri(REAL aV)
{
   return UniteFile * lround_ni(aV/UniteFile);
}

long long INT  cOriMntCarto::ToStdInt(REAL aV)
{
    return lround_ni(aV/UniteFile);
}


cOriMntCarto::cOriMntCarto
(
     const std::string & aName
)
{
    FILE * aFp = ElFopen(aName.c_str(),"r");
    ELISE_ASSERT(aFp!=0,"cOriMntCarto Can't Open");

    LireTexte(aFp,"CARTO");
    mOrigine.x = StdLireIntAsReal(aFp);
    mOrigine.y = StdLireIntAsReal(aFp);
    mZoneLambert = (int) LireInt(aFp);
    mOrigine.y += mZoneLambert * 1e6;
    mSz.x = (int) LireInt(aFp);
    mSz.y = (int) LireInt(aFp);
    mResol.x = StdLireIntAsReal(aFp);
    mResol.y = StdLireIntAsReal(aFp);

    LireTexte(aFp,"MNT");
    mZ0      = StdLireIntAsReal(aFp);
    mResolZ  = StdLireIntAsReal(aFp);

// Je ne connais plus le sens de Z0
    ELISE_ASSERT(mZ0==0,"cOriMntCarto Z0!=0");
}

REAL cOriMntCarto:: ResolZ() const {return mResolZ; }
REAL cOriMntCarto:: ResolPlani() const
{
   return ElMin(mResol.x,mResol.y);
}


cOriMntCarto::cOriMntCarto
(
     Pt2dr Ori,
     INT   aZoneLambert,
     Pt2di aSz,
     Pt2dr aResol,
     REAL  aZ0,
     REAL  aResolZ
)  :
   mOrigine      (Ori),
   mZoneLambert  (aZoneLambert),
   mSz           (aSz),
   mResol        (aResol),
   mZ0           (aZ0),
   mResolZ       (aResolZ)
{
}

void cOriMntCarto::ToFile(const std::string & aName)
{
   FILE * aFP = ElFopen(aName.c_str(),"w");
   ELISE_ASSERT(aFP!=0,"Cant open file in cOriMntCarto::ToFile");
   fprintf(aFP,"CARTO\n");
    #if (ELISE_MinGW)
        fprintf(aFP,"%I64d %I64d\n",ToStdInt(mOrigine.x),ToStdInt(mOrigine.y));
    #else
        fprintf(aFP,"%lld %lld\n",ToStdInt(mOrigine.x),ToStdInt(mOrigine.y));
    #endif
   fprintf(aFP,"%d\n",mZoneLambert);
   fprintf(aFP,"%d %d\n",mSz.x,mSz.y);
    #if (ELISE_MinGW)
        fprintf(aFP,"%I64d %I64d\n",ToStdInt(mResol.x),ToStdInt(mResol.y));
    #else
        fprintf(aFP,"%lld %lld\n",ToStdInt(mResol.x),ToStdInt(mResol.y));
    #endif

   fprintf(aFP,"\nMNT\n");
    #if (ELISE_MinGW)
        fprintf(aFP,"%I64d %I64d\n",ToStdInt(mZ0),ToStdInt(mResolZ));
    #else
        fprintf(aFP,"%lld %lld\n",ToStdInt(mZ0),ToStdInt(mResolZ));
    #endif
   ElFclose(aFP);
}

Pt2dr cOriMntCarto::ToPix(Pt2dr aP) const
{
   aP = aP-mOrigine;
   return Pt2dr(aP.x/ mResol.x,-aP.y/mResol.y) ;
}


Pt2dr cOriMntCarto::TerrainToPix(Pt2dr aP) const
{
   ELISE_ASSERT(mZoneLambert==0,"cOriMntCarto::TerrainToPix");
   return ToPix(aP);
}

Pt3dr  cOriMntCarto::PixToTerrain(Pt3dr aP) const
{
   return Pt3dr
          (
                 aP.x * mResol.x + mOrigine.x,
                 (-aP.y) * mResol.y + mOrigine.y,
                 aP.z * mResolZ  + mZ0
          );
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
