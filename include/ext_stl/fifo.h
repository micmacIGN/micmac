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



#ifndef _ELISE_EXT_STL_FIFO
#define _ELISE_EXT_STL_FIFO

/*************************************************************/
/* SOME UTILS ON TAB                                         */
/*************************************************************/

template <class Type> class ElFilo
{
     protected :

        void del_tab()
        {
             for (INT kk=0; kk<_capa; kk++) _tab[kk] = Type();
             delete [] _tab;
        }

        void incr_capa()
        {
           Type * NewTab = new Type [2*_capa+1]; //G++11
           for (int kk=0; kk<_nb ; kk++)
                NewTab[kk] = _tab[kk];

            del_tab();
           _tab = NewTab;
           _capa = 2*_capa+1; // G++11
        }


        Type * _tab;
        int  _nb;
        int   _capa;

        ElFilo(const ElFilo<Type>&);


     public :

        typedef Type value_type;

        int   nb   ()  const {return _nb;}
        int   size   ()  const {return _nb;}
        bool  empty()  const {return _nb == 0;}
        INT     capa() const {return _capa;}
        void  clear() { _nb    = 0;}

        void set_nb(INT Nb)
        {
             ELISE_ASSERT(Nb<=_capa,"ElFilo::set_nb"); // HJMPD,  < devient <=
             _nb = Nb;
        }

        void set_capa(INT new_capa)
        {
             while (_capa<new_capa)
             {
                 incr_capa();
             }
        }

        ElFilo(int aCapa = 10)  :
            _tab       (new Type [aCapa]),
            _nb        (0),
            _capa      (aCapa)
        {
        }
        const Type &  operator [] (int KK)  const
        {
             ELISE_ASSERT((KK>=0)&&(KK<_nb),"out index in ElFilo<Type>::[]");
             return _tab[KK];
        }
        Type &  operator [] (int K)
        {
             ELISE_ASSERT((K>=0)&&(K<_nb),"out index in ElFilo<Type>::[]");
             return _tab[K];
        }

        Type  & top()
        {
             ELISE_ASSERT(_nb!=0,"empty  ElFilo<Type>::top");
             return  _tab[_nb-1];
        }
        Type  & back() { return  top();}
        Type  & front() 
	{
             ELISE_ASSERT(_nb!=0,"empty  ElFilo<Type>::top");
             return  _tab[0];
	}


        const Type  & top() const
        {
             ELISE_ASSERT(_nb!=0,"empty  ElFilo<Type>::top");
             return  _tab[_nb-1];
        }

        Type  & top(INT k)
        {
              ELISE_ASSERT((k>=0)&&(k<_nb),"out index in ElFifo<Type>::[]");
              return  _tab[_nb-1-k];
        }

        const Type  & top(INT k) const
        {
              ELISE_ASSERT((k>=0)&&(k<_nb),"out index in ElFifo<Type>::[]");
              return  _tab[_nb-1-k];
        }



        ~ElFilo() 
        { 
             del_tab();
        }

        Type  poplast()
        {
             ELISE_ASSERT(_nb!=0,"empty  ElFilo<Type>::[]");
             return _tab[--_nb];
        }

        void   pushlast(const Type & p)
        {
             if (_nb == _capa) incr_capa();
#if Compiler_Visual_6_0  
             _tab[_nb++] = (Type) p;
#else
	     _tab[_nb++] =  p;
#endif
        }

        Type * tab() {return _tab;}

     
        Type * first() {return _tab;}
        Type * end()  {return _tab+_nb;}
        const Type * first() const {return _tab;}
        const Type * end()  const {return _tab+_nb;}
        typedef Type * iterator;
        typedef const Type * const_iterator;
};




template <class Type> class ElFifo
{
     protected :

        void del_tab()
        {
             for (INT kk=0; kk<_capa; kk++) _tab[kk] = Type();
             delete [] _tab;
        }


        void incr_capa()
        {
           Type * NewTab = new Type [2*_capa];
           for (INT kk=0; kk<_nb ; kk++)
                NewTab[kk] = _tab[(kk+_begin)%_capa];

            del_tab();
           _tab = NewTab;
           _begin = 0;
           _capa *= 2;
        }


        Type * _tab;
        int  _nb;
        int   _capa;
        int   _begin;
        bool _is_circ;

        ElFifo(const ElFifo<Type>&);


     public :

        typedef Type value_type;

        int   nb   ()  const {return _nb;}
        unsigned int   size   ()  const {return _nb;}
        int   nb_interv ()  const {return _nb +_is_circ;}
        bool  empty()  const {return _nb == 0;}
        bool circ()    const {return _is_circ;}
        void set_circ(bool Circ)  {_is_circ = Circ;}
        INT     capa() const {return _capa;}
        void  clear() { _nb    = 0; _begin = 0;}

        Type * tab()
        {
             ELISE_ASSERT(_begin==0,"_begin !=0 in ElFifo<Type>::tab()");
             return _tab;
        }

        ElFifo(int aCapa = 1,bool Circ = false)  :
            _tab       (new Type [aCapa]),
            _nb        (0),
            _capa      (aCapa),
            _begin     (0),
            _is_circ   (Circ)
        {
        }
        ElFifo(const std::vector<Type> & V,bool Circ)  :
            _tab       (new Type [V.size()]),
            _nb        (0),
            _capa      ((int) V.size()),
            _begin     (0),
            _is_circ   (Circ)
        {
            for (INT aK=0; aK<INT(V.size()) ; aK++)
                push_back(V[aK]);
        }

        std::vector<Type> ToVect()
	{
             std::vector<Type> aV;
	     for (INT aK=0; aK<nb() ; aK++)
		     aV.push_back((*this)[aK]);
	     return aV;
	}

        const Type &  operator [] (int k)  const
        {

              if (_is_circ)
              {
                 ELISE_ASSERT(_nb>0,"empty  ElFifo<Type>::[]");
                 return _tab[(mod(k,_nb)+_begin)%_capa];
             }
             else
             {
                 ELISE_ASSERT((k>=0)&&(k<_nb),"out index in ElFifo<Type>::[]");
                 return _tab[(k+_begin)%_capa];
             }

        }
        Type &  operator [] (int k)
        {
              if (_is_circ)
              {
                 ELISE_ASSERT(_nb>0,"empty  ElFifo<Type>::[]");
                 return _tab[(mod(k,_nb)+_begin)%_capa];
             }
             else
             {
                 ELISE_ASSERT((k>=0)&&(k<_nb),"out index in ElFifo<Type>::[]");
                 return _tab[(k+_begin)%_capa];
             }

        }

		Type  & back()
        {
             ELISE_ASSERT(_nb!=0,"empty  ElFifo<Type>::top");
             return  _tab[(_begin+_nb-1)%_capa];
        }
        Type  & top() {return back();}
		Type & front() {return _tab[_begin];}

        const Type  & back() const
        {
             ELISE_ASSERT(_nb!=0,"empty  ElFifo<Type>::top");
             return  _tab[(_begin+_nb-1)%_capa];
        }
		const Type & front() const {return _tab[_begin];}
        const Type  & top() const {return back();}

        Type  & top(INT k)
        {
              if (_is_circ)
              {
                 ELISE_ASSERT(_nb>0,"empty  ElFifo<Type>::[]");
                 return  _tab[(_begin+mod(_nb-1-k,_nb))%_capa];
             }
             else
             {
                 ELISE_ASSERT((k>=0)&&(k<_nb),"out index in ElFifo<Type>::[]");
                 return  _tab[(_begin+_nb-1-k)%_capa];
             }
        }

        const Type  & top(INT k) const
        {
              if (_is_circ)
              {
                 ELISE_ASSERT(_nb>0,"empty  ElFifo<Type>::[]");
                 return  _tab[(_begin+mod(_nb-1-k,_nb))%_capa];
             }
             else
             {
                 ELISE_ASSERT((k>=0)&&(k<_nb),"out index in ElFifo<Type>::[]");
                 return  _tab[(_begin+_nb-1-k)%_capa];
             }
        }



        ~ElFifo() 
         { 
            del_tab();
         }

        Type     popfirst()
        {
             ELISE_ASSERT(_nb!=0,"empty  ElFifo<Type>::[]");
             Type res = _tab[_begin];
             _begin = (_begin+1)%_capa;
             _nb--;
             return res;
        }
		void pop_front() {_begin = (_begin+1)%_capa; _nb--;}
        Type  poplast()
        {
             ELISE_ASSERT(_nb!=0,"empty  ElFifo<Type>::[]");
             Type res = _tab[(_begin+_nb-1)%_capa];
             _nb--;
             return res;
        }
		void  pop_back() {_nb--;}

        void   push_back(const Type & p)
        {
             if (_nb == _capa) incr_capa();
             _tab[(_begin+_nb++)%_capa] = p;
        }
        void   pushlast(const Type & p){push_back(p);}


        void   push_front(const Type & p)
        {
            if (_nb == _capa) incr_capa();
            _nb ++;
            _begin--;
            if (_begin<0)
               _begin += _capa;
            _tab[_begin] = p;
        }
        void   pushfirst(const Type & p){push_front(p);}

        void   push(const Type & p,bool in_last)
        {
              if (in_last) 
                 pushlast(p);
              else
                 pushfirst(p);
        }
};




template <class Type> class ElPartition;
template <class Type> class ElSubFilo
{
     public :
         typedef Type value_type;
         friend class ElPartition<Type>;

         INT size() const { return _i2-_i1;}
         INT nb() const { return size();}
         Type & operator [] (int k) {return (*_f)[k+_i1];}
         const Type & operator [] (int k) const {return (*_f)[k+_i1];}
         const Type & top (INT k=0) const {return (*_f)[_i2-1-k];}
         Type * tab() {return (*_f).tab()+_i1;}

         ElSubFilo() : _f(0), _i1(0), _i2(0) {}


         void * AdrFilo() {return _f;}
         INT    I1 () {return _i1;}
         INT    I2 () {return _i2;}

          std::vector<Type> ToVect()
	  {
             std::vector<Type> aV;
	     for (INT aK=0; aK<nb() ; aK++)
		     aV.push_back((*this)[aK]);
	     return aV;
	  }

      private :
         ElSubFilo(ElFilo<Type> & f,INT i1,INT i2) : _f(&f), _i1(i1), _i2(i2) {}

         ElFilo<Type> *    _f;
         INT               _i1;
         INT               _i2;
};
template <class Type> class ElPartition
{
     public :
           typedef Type value_type;
           INT nb() const  {return  _adr.nb()-1;}
           ElSubFilo<Type> operator[](INT k) 
           {
               ELISE_ASSERT((k>=0) && (k<nb()),"ElPartition::[](int)");
               return ElSubFilo<Type>(_f,_adr[k],_adr[k+1]);
           }
           ElSubFilo<Type> top(INT k=0) {return (*this)[nb()-1 -k];}

           void add(const Type & v) {_f.pushlast(v);}
           void close_cur()
           {
                _adr.pushlast(_f.nb());
           }

           void  remove_cur()
           {
                 while (_f.nb() >_adr.top())
                       _f.poplast();
           }

           ElPartition()         {_adr.pushlast(0);}
           void clear()          {_f.clear();_adr.clear(); _adr.pushlast(0);}
           ElFilo<Type> & filo() {return _f;}

      private  :
           ElFilo<Type>    _f;
           ElFilo<INT>     _adr;
};

template <class T1,class T2,class T3>
void append(T1 & out,const T2 & in,T3 f)
{
     for(INT k=0; k<in.nb(); k++)
        out.pushlast(f(in[k]));
}
template <class T1,class T2,class T3>
void copy_on(T1 & out,const T2 & in,T3 f)
{
     out.clear();
     append(out,in,f);
}


template <class T1,class T2>
void append(T1 & out,const T2 & in)
{
     for(INT k=0; k<in.nb(); k++)
        out.pushlast(in[k]);
}
template <class T1,class T2>
void copy_on(T1 & out,const T2 & in)
{
     out.clear();
     append(out,in);
}




template <class T1> void ElReverse(T1 & F)
{
    for (INT k1=0, k2=F.nb()-1; k1<k2 ; k1++,k2--)
        ElSwap(F[k1],F[k2]);
}              

#define IndexNoFind -0xfffffff

template <class T1,class T2> 
         INT  ElFind(T1 & F,T2 & v)
{
      for (INT k=0; k<F.nb(); k++)
          if (F[k] == v)
             return k;
              
      return IndexNoFind;
}



#endif /* ! _ELISE_EXT_STL_FIFO */








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
