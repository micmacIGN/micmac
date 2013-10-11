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



#ifndef _ELISE_ALLOCATION_H
#define _ELISE_ALLOCATION_H



class  Mcheck
{
      public :
         void * operator new    (size_t sz);
         void operator delete   (void * ptr) ;

      private :
      
       // to avoid use 
         void * operator new []  (size_t sz);
         void operator delete [] (void * ptr) ;
};

template <const INT NBB> class  ElListAlloc
{
     public :
         void * get()
         {
              if (_l.empty())
                 _l.push_back(malloc(NBB));
               void * res = _l.front();
               _l.pop_front();
               return res;
         }
         void  put(void * v)
         {
              _l.push_front(v);
         }
         void purge()
         {
             while (! _l.empty())
             {
                 free(_l.front());
                 _l.pop_front();
             }
         }

     private :
          ElSTDNS list<void *> _l;
};


/*******************************************************************/
/*                                                                 */
/*        Classes for garbage collection                           */
/*                                                                 */
/*******************************************************************/

/*
    Class RC_Object :

         References Counting Object
*/


class RC_Object : public Mcheck
{

   friend void decr_ref(class Object_cptref * oc);
   friend class PRC0;


   protected :

       RC_Object();
       virtual ~RC_Object();

    //---- data ----
       union
       {
           int            cpt_ref;
           RC_Object * next;
       }    _d;

   private :

      

   // declared as static so that they can be called with 0

      static void decr_ref(RC_Object *);
   public :
      static void incr_ref(RC_Object *);

};

/*
    Class PRC0 :

         Pointer to References Counting Object
*/

class PRC0
{

   public :


       // big three :

       PRC0 (const PRC0&);
       ~PRC0(void) ;
       void operator=(const PRC0 &p2);



       void * ptr(){return _ptr;}
      PRC0 (RC_Object *) ;

   protected :

      inline PRC0(){_ptr=0;};

    //-------- data ----------

        RC_Object     * _ptr;

   private :
         static  void decr_ref(RC_Object * oc);

      // just to avoid use  :
         // PRC0 * operator &();
};




#endif //  ! _ELISE_ALLOCATION_H 


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
