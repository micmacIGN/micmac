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

bool DebugCopy = false;

Fonc_Num::~Fonc_Num() {}
Output::~Output() {}
Flux_Pts::~Flux_Pts() {}

static const INT nb_max_level_copy = 10;
static INT nb_level_copy = -1;

static INT    line_err[nb_max_level_copy]  ;
static const char * file_err[nb_max_level_copy]  ;

void message_copy_where_error()
{
     if ((nb_level_copy >=0) && (nb_level_copy < nb_max_level_copy))
     {
        cout 
             << "********************************************************\n"
             << "**      (YOUR)  LOCATION :                              \n"
             << "**                                                      \n"
             << "** Error probably occured in a call to ELISE_COPY \n" 
             << "**         at line : " << line_err[nb_level_copy]  << "\n"
             << "**         of file : " << file_err[nb_level_copy]  << "\n"
             << "********************************************************\n";

         AddMessErrContext
         (
               std::string("Error in ELISE_COPY, called at line ") 
             + ToString(line_err[nb_level_copy]) 
             + " of file " 
             + file_err[nb_level_copy]
         );
      }
          
}

void ElCopy
     (
         Flux_Pts flux_pts,
         Fonc_Num fonc_num,
         Output output,
         INT line,
         const char * file
     )
{
    ElisePenseBete();

    Elise_tabulation::init();
   
    nb_level_copy++;
    if (nb_level_copy < nb_max_level_copy)
    {
       line_err[nb_level_copy] = line;
       file_err[nb_level_copy] = file;
    }

    Flux_Pts_Computed *flx  = flux_pts.compute(Arg_Flux_Pts_Comp());
    if (DebugCopy) BRKP;
    Fonc_Num_Computed *fonc = fonc_num.compute(Arg_Fonc_Num_Comp(flx));
    if (DebugCopy) BRKP;
    Output_Computed    *out = output.compute(Arg_Output_Comp(flx,fonc));

    const Pack_Of_Pts * pck;
    INT CPck=0;
    while ((pck = flx->next()))
    {
          CPck++;
          const Pack_Of_Pts *  vals = fonc->values(pck);
          OLD_BUG_CARD_VAL_FONC(pck,vals);
          out->update(pck,vals);
    }

    delete flx;
    delete fonc;
    delete out; 
    nb_level_copy--;
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
