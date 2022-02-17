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



#ifndef _ELISE_SMART_POINTEUR_H
#define _ELISE_SMART_POINTEUR_H



#if (DEBUG_INTERNAL)
class Memory_Counter
{
   public :
       void show(Memory_Counter);
       void verif(Memory_Counter);
       Memory_Counter(const char *name) ;
       Memory_Counter(void);

       const char * _name;
       INT _sum_call;
       INT _sum_size;
       INT _sum_ptr;

#if (CPP_OPTIMIZE)
       void * add(void * adr)
       {
            return add_sub_oks(adr,1,1);
       }

       void sub(void * adr)
       {
            add_sub_oks(adr,1,-1);
       }
#else
       void * add(void * adr);
       void sub(void * adr);
#endif

       void * add_sub_oks(void * adr,INT sz,INT sign);
};

extern Memory_Counter MC_OKS;
extern Memory_Counter MC_CPTR;
extern Memory_Counter MC_NTAB;
extern Memory_Counter MC_NEW_ONE;
extern Memory_Counter MC_TAB_USER;  // for the memory under the responsability of user

#define NB_MEMO_COUNTER 5
typedef  Memory_Counter  All_Memo_counter[NB_MEMO_COUNTER];
void stow_memory_counter(All_Memo_counter &);
void verif_memory_state(All_Memo_counter);

#define __DEBUG_MEM
#ifdef __DEBUG_MEM
	template <class T>
	T* __check_allocation( size_t nbObj )
	{
		T *res = NULL;
		try{
			res = new T [nbObj];
		}
		catch ( const std::bad_alloc & ){
			cerr << "__check_allocation : bad_alloc: " << nbObj << " objects of type [" << typeid(T).name() << ']' << endl;
			cerr.flush();
			cin.get();
			exit(EXIT_FAILURE);
		}
		catch ( const std::exception & ){
			cerr << "__check_allocation : fuck ><" << endl;
			cerr.flush();
			exit(EXIT_FAILURE);
		}
		if ( res==NULL ){
			cerr << "__check_allocation : NULL pointer" << endl;
			cerr.flush();
			exit(EXIT_FAILURE);
		}
		return res;
	}
	
	template <class T>
	T* __check_allocation1()
	{
		T *ptr = NULL;
		try{
			ptr = new T;
		}
		catch ( const std::bad_alloc & ){
			cerr << "__check_allocation : bad_alloc" << endl;
			cerr.flush();
			exit(EXIT_FAILURE);
		}
		catch ( const std::exception & ){
			cerr << "__check_allocation : fuck ><" << endl;
			cerr.flush();
			exit(EXIT_FAILURE);
		}
		if ( ptr==NULL ){
			cerr << "__check_allocation : NULL pointer" << endl;
			cerr.flush();
			exit(EXIT_FAILURE);
		}

		return ptr;
	}

	#define SAFE_ALLOC(Type,sz) (__check_allocation<Type>(sz))
	#define SAFE_ALLOC1(Type) (__check_allocation1<Type>())
#else
	#define SAFE_ALLOC(Type,sz) (new Type [sz])
	#define SAFE_ALLOC1(Type) (new Type)
#endif

#define ADD_MEM_COUNT(MC,adr,sz) (MC.add_sub_oks(adr,sz,1))
#define SUB_MEM_COUNT(MC,adr,sz) (MC.add_sub_oks(adr,sz,-1))

#else   /*  DEBUG_INTERNAL */
	//  A class to allow some check-sum on memory alloc and desalloc
	#define SAFE_ALLOC(Type,sz) (new Type [sz])
	#define SAFE_ALLOC1(Type) (new Type)

	#define MC_OKS
	#define MC_CPTR
	#define MC_NTAB
	#define MC_NEW_ONE
	#define MC_TAB_USER

	#define ADD_MEM_COUNT(MC,adr,sz) (adr)
	#define ADD_MEM_COUNT2(MC,adr,sz) (adr)
	#define SUB_MEM_COUNT(MC,adr,sz) (adr)

	typedef  void * All_Memo_counter;
	void stow_memory_counter(All_Memo_counter &);
	void verif_memory_state(All_Memo_counter);
#endif /*  DEBUG_INTERNAL */

void * std_dup(void * out,const void * in,INT sz_nb);
char * std_ch_dup(const char * ch);


/*
#define STD_NEW_TAB(nb,Type)\
((Type *) ADD_MEM_COUNT(MC_NTAB,(new Type [nb]),1))
*/
#define STD_NEW_TAB(nb,Type)\
((Type *) ADD_MEM_COUNT( MC_NTAB, SAFE_ALLOC(Type,nb), 1 ))

#define STD_DELETE_TAB(p)\
      ((void)SUB_MEM_COUNT(MC_NTAB,p,1) ,  (delete  [] (p)))

/*
#define STD_NEW_TAB_DUP(val,Type,nb)\
((Type *) ADD_MEM_COUNT(MC_NTAB,std_dup(new Type [nb],val,sizeof(Type)*nb),1))
*/
#define STD_NEW_TAB_DUP(val,Type,nb)\
((Type *) ADD_MEM_COUNT(MC_NTAB,std_dup( SAFE_ALLOC(Type,nb),val,sizeof(Type)*nb),1))

/*
#define STD_NEW_TAB_USER(nb,Type)\
((Type *) ADD_MEM_COUNT( MC_TAB_USER,new Type [nb],1) )
*/
#define STD_NEW_TAB_USER(nb,Type)\
((Type *) ADD_MEM_COUNT( MC_TAB_USER, SAFE_ALLOC(Type,nb), 1 ) )


#define STD_DELETE_TAB_USER(p)\
      ((void)SUB_MEM_COUNT(MC_TAB_USER,p,1) ,  (delete  [] (p)))





/********************** smart_pointer.h *****************************

      Ce fichier est un essai de definition d'un classe
   de "smart pointeur" qui doivent permettre, dans les phases de
   debugages d'attraper la plupart des erreurs courrantes dans
   la gestion de la memoire.

      L'objectif est que, une fois le flag de debug on retrouve
   exactement le meme comportement qu'avec des pointeurs "normaux".

*********************************************************************/

#if DEBUG_SMART_POINTEUR


template <class > class Smart_Pointer;



/********************************************************/

/*    Remarques generales sur les "Smart_Pointer" :

  [1]
      Cela semble un peu dommage d'iplanter toute la genericite  a base de template,
      alors que la seule partie du code ou le "Type" soit utilise est pour
      "caster" le resultat de "*" et "[]". Theoriquement, a peu pres toutes les
      methode de la classe template "Smart_Pointer" pourraient etre definies
      dans une classe de base Smart_Pointer_Gen. En partique ca ne marcherait pas
      car :

         - l'operateur "=" est le seul qui n'est pas herite;
         - il est pobable (a verifier) que constructeur et desctructeur ne sont pas
           herites;

         - un operateur comme "+" (dans "p+3"), est totalement independant du type
           d'objet reference, sauf pour le type de retour du pointeur. Donc, si on
           voulait l'ecrire dans une classe de bases, il faudrait effectuer des tas
           de conversion via des macros.

      Peut-etre qu'une inplantation sous la forme "has a" ou lieu de "is a" 
      permettraient de mettre plus de code en commun ? C'est a dire :
         - une clas Smart_Pointer_Gen
         - les classe template possedent  un champs Smart_Pointer_Gen.
      Bof, bof ...


*/

class Memory_seg;


class Smart_Pointer_Gen
{
     public    :



     protected :

        Smart_Pointer_Gen(void);
        Smart_Pointer_Gen(INT4);

      // utilitaires  

         void bit_copy(const Smart_Pointer_Gen & p2);
            // renvoie le debut du segment memoire si le pointeur est initialise, non nul et
            // que la zone est non liberee. Sinon genere une "Fatale_Erreur".
         void * smp_mem_nn(void) const;
            // == smp_mem_nn mais ne genere pas d'erreur avec le pointeur null
         void * smp_mem_nul(void) const;

         void incr_ref(void) const ;
         void decr_ref(void) ;

      // champs

         class Memory_Seg * _zone;
         INT4         _offset; 


     private :
};



/**********************************************************************/
/**********************************************************************/
 
template <class Type> class Smart_Pointer  : public Smart_Pointer_Gen
{
     public :

        void delete_tab(void);
        void delete_unaire(void);



         // constructeurs + destructeur
              // Pour memoriser que le pointeur est non initialise
         Smart_Pointer(void): Smart_Pointer_Gen(){};

             // ce constructeur ne sert qu'a creer le pointeur null
             // pour que, en mode debug ou non debug, on puisse toujours
             // ecrire  UChar_p (0)
         Smart_Pointer (INT4 nb) : Smart_Pointer_Gen(nb){};
             // static car l'expansion de la macro new tab doit appeler
             // la methode sans qu'il y ait d'objet associe.
         static Smart_Pointer<Type>   new_tab(INT4);
         static Smart_Pointer<Type>   new_vect(INT4,INT4);

         static Smart_Pointer<Type>   new_val(void);
         ~Smart_Pointer(void);

         friend Smart_Pointer<Type> ADRESS_OF(Type&);
         friend Smart_Pointer<Type> STAT_PTR(Type *,INT4 nb);
         Type * export(void);


         // copie + affectation
         Smart_Pointer<Type> & operator=(const Smart_Pointer<Type> &);
         Smart_Pointer(const Smart_Pointer<Type> &);


         // derereferencement
         Type & operator * (void);
         Type & operator [] (INT4);
         Type * operator -> (void);

          //************ arithmetique ************************

              // dans 3 + Ptr, "+" ne peut etre definie que comme friend
              // et pas comme operateur. Par homogeneite on defini alors "+"
              // et "-" comme des friend. 

              // arithmetique simple
         friend  Smart_Pointer<Type> operator + (const Smart_Pointer<Type> &,INT4);
         friend  Smart_Pointer<Type> operator + (int, const Smart_Pointer<Type> &);
         friend  Smart_Pointer<Type> operator - (const Smart_Pointer<Type> &,INT4);
         friend  INT4 operator - (const Smart_Pointer<Type> &,
                                 const Smart_Pointer<Type> &);

              // arithmetique +=, -=
          Smart_Pointer<Type> & operator += (INT4);
          Smart_Pointer<Type> & operator -= (INT4);

              // arithmetique p++, ++p

           Smart_Pointer<Type> & operator ++ ();      // ++p
           Smart_Pointer<Type>   operator ++ (INT4);   // p++
           Smart_Pointer<Type> & operator -- ();      // --p
           Smart_Pointer<Type>   operator -- (INT4);   // p--

              // comparaison  <, >, <=, >=

           INT4  operator<  (const Smart_Pointer<Type> &);      
           INT4  operator<= (const Smart_Pointer<Type> &);      
           INT4  operator>  (const Smart_Pointer<Type> &);      
           INT4  operator>= (const Smart_Pointer<Type> &);      

              // egalite 

           INT4  operator== (const Smart_Pointer<Type> &);      
           INT4  operator!= (const Smart_Pointer<Type> &);      


              // le ! qui apparament fait partie des operateurs sur
              // les pointeurs, avec la semantique (! p) <=> (p == 0)

           INT4  operator! ();

            static CONST_STAT_TPL  Type _VAL_DEF;
            // why not " static const Type _VAL_DEF" ?
            // because of mystery bugs shown in file
            // "bugs_cpp/stat_const_class.C"

     protected :
     private :
         INT4  sh_ref(void);  // debug, put public when needed.
         INT4 offset(void){return _offset;}  // debug

};


/*************************************************************/
/*    SUPERIOR ORDER POINTER                                 */
/*     not implemented (specicifcally), see smart_pointeur.C */
/*     for some remarks.                                     */
/*************************************************************/



/* !!!!!

   The new_matr function, is rather strangely declared, but

      - it cannot be a member method (else there is an infinite recursion in
        template instantiation);

      - overloading cannot be resolved by return type; this is why
        there is a (useless) third argument of type "Type *";

*/

template <class Type> Smart_Pointer<Smart_Pointer<Type> >   new_matr(Pt2di,Pt2di,Type *);

template <class Type> void delete_matr (Smart_Pointer<Smart_Pointer<Type> >, Pt2di,Pt2di);


/* Les macros ou types a redefinir si DEBUG = 0 */



#define NEW_VECTEUR(x1,x2,type) Smart_Pointer<type>::new_vect(x1,x2)
#define NEW_MATRICE(p1,p2,type) new_matr(p1,p2,(type *) 0)
#define NEW_TAB(nb,Type)  (Smart_Pointer<Type>::new_tab(nb))
#define NEW_ONE(Type)         (Smart_Pointer<Type>::new_val())

#define DELETE_TAB(p)     ((p).delete_tab())
#define DELETE_VECTOR(v,x) DELETE_TAB((v)+(x))
#define DELETE_ONE(p)         ((p).delete_unaire())
#define DELETE_MATRICE(m,p1,p2)  delete_matr(m,p1,p2)

// #define EXPORT(p)  ((p).export())

#define U_INT1_Ptr   Smart_Pointer<U_INT1>
#define U_INT1_PPtr  Smart_Pointer<U_INT1_Ptr>
#define U_INT2_Ptr   Smart_Pointer<U_INT2>
#define INT4_Ptr     Smart_Pointer<INT4>
#define INT4_PPtr    Smart_Pointer<INT4_Ptr>
#define REAL8_Ptr    Smart_Pointer<REAL8>
#define REAL8_PPtr   Smart_Pointer<REAL8_Ptr>
#define Pt2di_Ptr    Smart_Pointer<Pt2di>

#define SMP(Un_Type)    Smart_Pointer<Un_Type>
#define SMPP(Un_Type)   Smart_Pointer<Smart_Pointer<Un_Type> >


#else   /* DEBUG_SMART_POINTEUR */


extern void * Elise_Calloc(size_t nmemb, size_t size);
extern void  Elise_Free(void *);



extern void **  alloc_matrice(const Pt2di p1,const Pt2di p2,const INT sz);
extern void ***  alloc_tab_matrice(const INT nb,
                                   const Pt2di p1,
                                   const Pt2di p2,
                                   const INT sz);
extern void *  alloc_vecteur(const int x1,const int x2,const int sz);
extern void  delete_vecteur(void * v,const int x1,const int sz);
extern void delete_matrice(void ** m, const Pt2di p1,const Pt2di p2,const int sz);
extern void delete_tab_matrice(void *** m,INT nb, const Pt2di p1,const Pt2di p2,const int sz);

#define NEW_VECTEUR(x1,x2,type) ((type *) alloc_vecteur((x1),(x2),sizeof(type)))
#define NEW_MATRICE(p1,p2,type) ((type **) alloc_matrice((p1),(p2),sizeof(type)))
#define NEW_MATRICE_ORI(x,y,type) (NEW_MATRICE(Pt2di(0,0),Pt2di(x,y),type))

#define NEW_TAB_MATRICE(nb,p1,p2,type)\
 ((type ***) alloc_tab_matrice((nb),(p1),(p2),sizeof(type)))

/*
#define NEW_TAB(nb,Type)\
((Type *) ADD_MEM_COUNT(MC_NTAB,(new Type [nb]),1))
*/
#define NEW_TAB(nb,Type)\
((Type *) ADD_MEM_COUNT(MC_NTAB,SAFE_ALLOC(Type,nb),1))

/*
#define NEW_ONE(Type)\
((Type *) ADD_MEM_COUNT(MC_NEW_ONE,(new Type),1))
*/
#define NEW_ONE(Type)\
((Type *) ADD_MEM_COUNT(MC_NEW_ONE,SAFE_ALLOC1(Type),1))

#define CLASS_NEW_ONE(Type,Arg)\
((Type *) ADD_MEM_COUNT(MC_NEW_ONE,(new Type Arg),1))
/*
#define CLASS_NEW_ONE(Type,Arg)\
((Type *) ADD_MEM_COUNT(MC_NEW_ONE,SAFE_ALLOC_ARG(Type,Arg),1))
*/

#if (DEBUG_INTERNAL)
    #define DELETE_TAB(p)  ((void)SUB_MEM_COUNT(MC_NTAB,p,1) ,  (delete  [] (p)))
    #define DELETE_ONE(p)      (SUB_MEM_COUNT(MC_NEW_ONE,p,1)  ,   (delete  (p)))
#else
    #define DELETE_TAB(p)  delete  [] (p)
    #define DELETE_ONE(p)  delete  (p)
#endif

#define DELETE_VECTOR(v,x) delete_vecteur((void *)(v),(x),sizeof(*(v)))
#define DELETE_MATRICE(m,p1,p2)\
         delete_matrice((void **)(m),(p1),(p2),sizeof(**(m)))

#define DELETE_TAB_MATRICE(m,nb,p1,p2)\
         delete_tab_matrice((void ***)(m),(nb),(p1),(p2),sizeof(***(m)))

#define DELETE_MATRICE_ORI(m,x,y) DELETE_MATRICE(m,Pt2di(0,0),Pt2di(x,y))

// #define EXPORT(p)  (p)

#define ADRESS_OF(val)    (&(val))
#define STAT_PTR(val,nb)    (val)

typedef  U_INT1 *     U_INT1_Ptr; 
typedef  U_INT1 **    U_INT1_PPtr; 
typedef  U_INT2 *     U_INT2_Ptr; 
typedef  INT4 *       INT4_Ptr; 
typedef  INT4 **      INT4_PPtr; 
typedef  REAL8 *      REAL8_Ptr; 
typedef  REAL8 **     REAL8_PPtr; 
typedef  Pt2di *      Pt2di_Ptr; 

#define SMP(Un_Type)    Un_Type *
#define SMPP(Un_Type)   Un_Type **


template <class Type> Type * new_vecteur_init(INT x0,INT x1,Type v);


#endif  /* DEBUG_SMART_POINTEUR */







#endif /*! _ELISE_SMART_POINTEUR_H */


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
