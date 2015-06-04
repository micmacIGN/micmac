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

void ElisePenseBete()
{
   static bool Deja = false;
   if (Deja)
      return;
   Deja = true;
   std::string aName = "data/PenseBete.txt";
   if (! ELISE_fp::exist_file(aName))
     return;
   ELISE_fp  aFile(aName.c_str(),ELISE_fp::READ);
   string aBuf; //char aBuf[tBuf]; TEST_OVERFLOW
   bool eof=false;
   cout << "##################################\n";
   while (! eof)
   {
       if ( aFile.fgets(aBuf,eof)) //if ( aFile.fgets(aBuf,tBuf,eof)) TEST_OVERFLOW
          cout << aBuf << "\n";
   }
   cout << "##################################\n";
   aFile.close();
}

Pt2dr TAB_CornerPix[4] =
      {
	      Pt2dr(0.5,0.5),
	      Pt2dr(-0.5,0.5),
	      Pt2dr(-0.5,-0.5),
	      Pt2dr(0.5,-0.5)
      };

Pt2di VOIS_9[9] =
      {
          Pt2di( -1, -1) ,
          Pt2di(  0, -1) ,
          Pt2di(  1, -1) ,
          Pt2di( -1,  0) ,
          Pt2di(  0,  0) ,
          Pt2di(  1,  0) ,
          Pt2di( -1,  1) ,
          Pt2di(  0,  1) ,
          Pt2di(  1,  1) 
      };

Pt2di TAB_9_NEIGH[9] =
      {
          Pt2di( 1, 0) ,
          Pt2di( 1, 1) ,
          Pt2di( 0, 1) ,
          Pt2di(-1, 1) ,
          Pt2di(-1, 0) ,
          Pt2di(-1,-1) ,
          Pt2di( 0,-1) ,
          Pt2di( 1,-1) ,
          Pt2di( 0, 0) 
      };

int  PdsGaussl9NEIGH[9] = { 2,1,2,1,2,1,2,1,4 };



Pt2di TAB_8_NEIGH[16] =
      {
          Pt2di( 1, 0) ,
          Pt2di( 1, 1) ,
          Pt2di( 0, 1) ,
          Pt2di(-1, 1) ,
          Pt2di(-1, 0) ,
          Pt2di(-1,-1) ,
          Pt2di( 0,-1) ,
          Pt2di( 1,-1) ,

          Pt2di( 1, 0) ,
          Pt2di( 1, 1) ,
          Pt2di( 0, 1) ,
          Pt2di(-1, 1) ,
          Pt2di(-1, 0) ,
          Pt2di(-1,-1) ,
          Pt2di( 0,-1) ,
          Pt2di( 1,-1) 
      };

Pt2di TAB_5_NEIGH[5] =
      {
          Pt2di( 1, 0) ,
          Pt2di( 0, 1) ,
          Pt2di(-1, 0) ,
          Pt2di( 0,-1) ,
          Pt2di( 0, 0) 
      };


Pt2di TAB_4_NEIGH[8] =
      {
          Pt2di( 1, 0) ,
          Pt2di( 0, 1) ,
          Pt2di(-1, 0) ,
          Pt2di( 0,-1) ,

          Pt2di( 1, 0) ,
          Pt2di( 0, 1) ,
          Pt2di(-1, 0) ,
          Pt2di( 0,-1) 
      };

INT TAB_8_FREEM_SUCC_TRIG[8] = {1,2,3,4,5,6,7,0};
INT TAB_4_FREEM_SUCC_TRIG[4] = {1,2,3,0};

INT TAB_8_FREEM_PREC_TRIG[8] = {7,0,1,2,3,4,5,6};
INT TAB_4_FREEM_PREC_TRIG[4] = {3,0,1,2};

INT TAB_8_FREEM_SYM[8] = {4,5,6,7,0,1,2,3};
INT TAB_4_FREEM_SYM[4] = {2,  3,  0,  1};

U_INT1 FLAG_FRONT_8_TRIGO[512];
U_INT1 FLAG_FRONT_4_TRIGO[512];

MAT_CODE_FREEM MAT_CODE_8_FREEM = {
                                       {5, 6,7},
                                       {4,-1,0},
                                       {3, 2,1}
                                  };

MAT_CODE_FREEM MAT_CODE_4_FREEM = {
                                       {-1, 3,-1},
                                       {2, -1, 0},
                                       {-1, 1,-1}
                                  };


INT compute_freem_code
    (
         MAT_CODE_FREEM & m,
         Pt2di                  p
    )
{
   Elise_tabulation::init();
    ASSERT_INTERNAL
    (
        (p.x>=-1) && (p.x<=1) && (p.y>=-1) && (p.y<= 1),
        "Error in compute_freem_code"
    );
    INT code = m[p.y+1][p.x+1];

    ASSERT_INTERNAL
    (
        code != -1,
        "Error in compute_freem_code"
    );
    return code;
}


INT freeman_code(Pt2di p)
{
    Elise_tabulation::init();

    if ( (p.x>=-1) && (p.x<=1) && (p.y>=-1) && (p.y<= 1))
       return  MAT_CODE_8_FREEM[p.y+1][p.x+1];
    else
       return -2;
}




bool Elise_tabulation::_deja = false;

void Elise_tabulation::_init()
{
    Tabul_Bits_Gen::init_tabul_bits();
}


class ELISE_GLOB_TABULATION
{
      public  :
         ELISE_GLOB_TABULATION();
         static ELISE_GLOB_TABULATION _THE_ONE;

         bool flag_opb3_positionne(INT flag,INT x,INT y)
         {
              return  (flag&(1<<(y+1+3*(x+1)))) != 0;
         }

         void init_front_opb3
         (
             U_INT1    (&tab)[512],
             INT *      succ,
             INT *      pred,
             bool       v8
         );

};



void ELISE_GLOB_TABULATION::init_front_opb3
     (
         U_INT1    (&tab)[512],
         INT *      succ,
         INT *      pred,
         bool       v8
     )
{
   int freem_to_opb3[8] = {7,8,5,2,1,0,3,6};

   for (INT  flag_opb3 = 0 ; flag_opb3 < 512 ; flag_opb3++)
       if (! flag_opb3_positionne(flag_opb3,0,0))
       {
             tab[flag_opb3] = 0;
       }
       else
       {
             INT flag_freem = 0;
			INT k;

             for ( k=0; k<8 ; k++)
                 if (flag_opb3&(1<<freem_to_opb3[k]))
                    flag_freem |= (1<<k);

             tab[flag_opb3] = 0;
             for ( k=0; k<8 ; k+=2 )
                 if ( 
                              (flag_freem & (1<<k))
                       &&  (! (flag_freem & (1<<pred[k])))
                       &&  (! (flag_freem & (1<<pred[pred[k]])))
                    )
                    tab[flag_opb3] |= (1<<k);
             for ( k=1; k<8 ; k+=2 )
                 if ( 
                              (flag_freem & (1<<k))
                       &&  (! (flag_freem & (1<<pred[k])))
                       &&  (v8 ||  (flag_freem & (1<<succ[k])))
                    )
                    tab[flag_opb3] |= (1<<k);
       }

}




ELISE_GLOB_TABULATION::ELISE_GLOB_TABULATION()
{
     ElisePenseBete();
     init_front_opb3
     (
         FLAG_FRONT_8_TRIGO,
         TAB_8_FREEM_SUCC_TRIG,
         TAB_8_FREEM_PREC_TRIG,
         true
     );

     init_front_opb3
     (
         FLAG_FRONT_4_TRIGO,
         TAB_8_FREEM_SUCC_TRIG,
         TAB_8_FREEM_PREC_TRIG,
         false
     );
}

ELISE_GLOB_TABULATION  ELISE_GLOB_TABULATION::_THE_ONE;


/*******************************************************/
/*                                                     */
/*      template <Type class> File_Tabulated           */
/*                                                     */
/*******************************************************/

template <class Type> File_Tabulated<Type>::File_Tabulated
                      (
                            const char * name
                      )   :
                         _ptr (0),
                         _name (name)
{
}

template <class Type> const Type * File_Tabulated<Type>::ptr()
{
    if (! _ptr)
    {
       INT nb = sizeofile(_name);

       ELISE_fp fp(_name,ELISE_fp::READ);
       _ptr = NEW_TAB_FOR_EVER(U_INT1,nb);
       fp.read(_ptr,sizeof(U_INT1),nb);
       fp.close();
    }
    return _ptr;
}

template class  File_Tabulated<U_INT1>;



Config_Freeman_Or::Config_Freeman_Or(bool v8,bool trigo) :

      _pts     (v8?TAB_8_NEIGH:TAB_4_NEIGH),
      _nb_pts  (v8?8:4),
      _succ    (v8?TAB_8_FREEM_SUCC_TRIG:TAB_4_FREEM_SUCC_TRIG),
      _prec    (v8?TAB_8_FREEM_PREC_TRIG:TAB_4_FREEM_PREC_TRIG),
      _sym     (v8?TAB_8_FREEM_SYM      :TAB_4_FREEM_SYM),
      _mat_code (v8?&MAT_CODE_8_FREEM:&MAT_CODE_4_FREEM)
{
     if(!trigo)
       ElSwap(_succ,_prec);
}

std::vector<Pt3di> DirCube(int aFlag)
{
    std::vector<Pt3di> aRes;
    for (int anX=-1 ; anX<=1 ; anX++)
    {
        for (int anY=-1 ; anY<=1 ; anY++)
        {
             for (int aZ=-1 ; aZ<=1 ; aZ++)
             {
                 int aSom = ElAbs(anX) +  ElAbs(anY) +  ElAbs(aZ);
                 Pt3di aP(anX,anY,aZ);
                 if (aFlag & (1<<aSom))
                    aRes.push_back(aP);
             }
        }
    }
    return aRes;
}

const std::vector<Pt3di> &  Dir6Cube()
{
     static std::vector<Pt3di> aRes = DirCube(2);
     return aRes;
}
const std::vector<Pt3di> &  Dir8Cube()
{
     static std::vector<Pt3di> aRes = DirCube(8);
     return aRes;
}
const std::vector<Pt3di> &  Dir14Cube()
{
     static std::vector<Pt3di> aRes = DirCube(2|8);
     return aRes;
}
const std::vector<Pt3di> &  Dir26Cube()
{
     static std::vector<Pt3di> aRes = DirCube(2|4|8);
     return aRes;
}





/*
Pt3di *  Dir8Cube(int & aNb);
Pt3di *  Dir14Cube(int & aNb);
Pt3di *  Dir26Cube(int & aNb);
*/


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
