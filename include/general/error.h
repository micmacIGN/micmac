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



/*******************************************************/
/*                                                     */
/*   ASSERTION.                                        */
/*                                                     */
/*   Will replace all the macro ASSERT* when I will    */
/*  get enouh time to  reparse all the code            */
/*                                                     */
/*******************************************************/

#pragma once



class ElEM   // Elise Erreur Message
{
      friend class Elise_Pile_Mess_0;

      public :

         inline ElEM(REAL16 i)
         {
               _type   = _real;
               _data.r = i;
         }
         inline ElEM(INT i)
         {
               _type   = _int;
               _data.i = i;
         }

         inline ElEM(REAL  r)
         {
               _type   = _real;
               _data.r = r;
         }

         inline ElEM(const Box2di  &b)
         {
               _type   = _box_2di;
               _data.box = &b;
         }

         inline ElEM(Pt2di  &pt)
         {
               _type   = _pt2di;
               _data.pt = &pt;
         }

         inline  ElEM(const char * s)
         {
               _type   = _string;
               _data.s = s;
         }

         inline  ElEM(const class Pack_Of_Pts *  pack,INT k)
         {
               _type      = _pt_pck;
               _data.pack = pack;
               _data_2.i  = k;
         }

         inline  ElEM(const INT *  pts,INT d)
         {
               _type      = _tab_int;
               _data.Pi = pts;
               _data_2.i  = d;
         }

         inline  ElEM(const REAL *  pts,INT d)
         {
               _type      = _tab_real;
               _data.Pr = pts;
               _data_2.i  = d;
         }




      private  :

         ElEM() {}; // for Elise_Pile_Mess_0::_stack
          void  mes_el() const ;

          typedef enum
          {
              _int,
              _real,
              _string,
              _pt_pck,
              _tab_int,
              _tab_real,
              _box_2di,
              _pt2di
          }  _type_of;

          _type_of _type;

          union
          {
             INT       i;
             REAL      r;
             const class Pack_Of_Pts *   pack;
             const char *    s;
             const INT *    Pi;
             const REAL *    Pr;
             const Box2di    *  box;
             const Pt2di *      pt;
          }  _data;

          union
          {
             INT       i;
          }  _data_2;


};

extern bool TheExitOnBrkp;
extern int  GetCharOnBrkp();

class Elise_Pile_Mess_N
{
    friend class Elise_Pile_Mess_0;
    public :
         inline  Elise_Pile_Mess_N
                 operator << (ElEM  m);

    private :
         Elise_Pile_Mess_N();
         static  Elise_Pile_Mess_N _the_one;
};

class Elise_Pile_Mess_0
{
     friend class Elise_Assertion;
     friend class Elise_Pile_Mess_N;

     public :
         inline  Elise_Pile_Mess_N
                 operator << (ElEM  m)
         {
                 _nb = 0;
                 _stack[_nb++] = m;
                 return Elise_Pile_Mess_N::_the_one;
         }

     private :

          static void display(const char *);
          
          static INT _nb;
          static ElEM _stack[100];
};
extern  Elise_Pile_Mess_0 EEM0;

inline  Elise_Pile_Mess_N
        Elise_Pile_Mess_N::operator << (ElEM  m)
{
       Elise_Pile_Mess_0::_stack[Elise_Pile_Mess_0::_nb++] = m;
       return _the_one;
}



class Elise_Assertion
{
      public :

          static void unactive_user();
          Elise_Assertion(const char *);

          inline void ElAssert(bool expr,Elise_Pile_Mess_N)
          {
                 if (_active && (! expr))
                    Elise_Pile_Mess_0::display(_kind_of);
                   
          }
          inline void error(Elise_Pile_Mess_N)
          {
               if (_active)
                  Elise_Pile_Mess_0::display(_kind_of);
          }
          bool active() const {return _active;}

      private :

          bool   _active;
          const char * _kind_of;
};


extern Elise_Assertion Tjs_El_User;
extern Elise_Assertion El_User_Dyn;
extern Elise_Assertion El_Internal;


class cElWarning
{
    public :
         void AddWarn
         (
            const std::string &  aMes,
            int                  aLine,
            const std::string &  aFile
         );

         static cElWarning  BehindCam;
         static cElWarning  FocInxifAndMM;
         static cElWarning  CamInxifAndMM;
         static cElWarning  GeomIncompAdaptF2Or;
         static cElWarning  GeomPointTooManyMeasured;
         static cElWarning  OrhoLocOnlyXCste;
         static cElWarning  ToVerifNuage;
         static cElWarning  TrueRot;
         static cElWarning  ScaleInNuageFromP;

         static void ShowWarns(const std::string & aFile);
    private :

         cElWarning(const std::string & aName);
         cElWarning(const cElWarning &); // N.I.
         void ShowOneWarn(FILE *);

         static std::vector<cElWarning  *> mWarns;
         static int                        mNbTot;

         std::string   mName;
         int           mNbWarns;
         std::string   mMes;
         int           mLine;
         std::string   mFile;
};



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
