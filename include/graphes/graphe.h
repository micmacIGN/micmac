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

#ifndef _ELISE_GRAPHES_GRAPHE_H
#define _ELISE_GRAPHES_GRAPHE_H

template <class T> class ElTabDyn;
template <class AttrSom,class AttrArc> class ElGraphe;
template <class AttrSom,class AttrArc> class ElArc;
template <class AttrSom,class AttrArc> class ElSom;
template <class AttrSom,class AttrArc> class ElSubGraphe;
template <class AttrSom,class AttrArc> class ElSomIterator;
template <class AttrSom,class AttrArc> class ElArcIterator;

#define PRIVATE private

template <class T> class ElTabDyn 
{
    public :
      ElTabDyn() : _tvals (0), _nb(0) {}
      ~ElTabDyn();
      INT nb() { return _nb;}

      T & operator [] (int k)
      {
         ELISE_ASSERT((k>=0)&&(k<_nb),"ElTabDyn indexing");
         return _tvals[i0(k)][i1(k)][i2(k)];
      }
      void augment_index();
      void set_intexable_until(INT NB);


    private :

      T ** & t1(INT k) {return  _tvals [i0(k)]       ;}
      T *  & t2(INT k) {return  _tvals [i0(k)][i1(k)];}

      INT   i0(INT k){return  k/v2    ;}
      INT   i1(INT k){return (k/v1)%v1;}
      INT   i2(INT k){return  k%v1    ;}

        
      T *** _tvals;
      INT   _nb;

      enum
      {
            NB = 256,

            v1 = NB,
            v2 = v1*v1,
            v3 = v2*v1
      };
};


template <class AttrSom,class AttrArc> class ElSubGrapheSom
{
      public :
          typedef AttrSom ASom;
          typedef AttrArc AArc;
          typedef ElSom<AttrSom,AttrArc>          TSom;
          virtual bool   inS(TSom &)  {return true;}
	  virtual ~ElSubGrapheSom() {}
};

template <class AttrSom,class AttrArc> 
         class ElEmptySubGrapheSom : public  ElSubGrapheSom<AttrSom,AttrArc>
{
      public :
          typedef ElSom<AttrSom,AttrArc>          TSom;
          typedef AttrSom ASom;
          typedef AttrArc AArc;
          virtual bool   inS(TSom &)  {return false;}
};

template <class AttrSom,class AttrArc> 
         class ElSubGrapheSingleton : public ElSubGrapheSom<AttrSom,AttrArc>
{
      public :
          typedef ElSom<AttrSom,AttrArc>          TSom;
          typedef AttrSom ASom;
          typedef AttrArc AArc;
          virtual bool   inS(TSom & s2)  
          {
              return (&s2) == _s;
          }

          ElSubGrapheSingleton(TSom & s) : _s (&s) {}

      private :
          TSom  * _s;
};

template <class AttrSom,class AttrArc> 
         class ElSubGrapheInFilo : public ElSubGrapheSom<AttrSom,AttrArc>
{
      public :
          typedef ElSom<AttrSom,AttrArc>          TSom;
          typedef AttrSom ASom;
          typedef AttrArc AArc;
          virtual bool   inS(TSom & s)  
          {
              return (_flag>=0) && (s.flag_kth(_flag));
          }

          ElSubGrapheInFilo(ElFilo<TSom *> & ts) : 
               _ts   (ts) ,
               _flag (ts.nb()?ts[0]->gr().alloc_flag_som() : -1)
          {
               for (INT k=0; k<_ts.nb() ; k++)
                   _ts[k]->flag_set_kth_true(_flag);
          }

          virtual ~ElSubGrapheInFilo() 
          {
               for (INT k=0; k<_ts.nb() ; k++)
                   _ts[k]->flag_set_kth_false(_flag);
               if(_flag>=0)
                   _ts[0]->gr().free_flag_som(_flag);
          }

      private :

          ElFilo<TSom *> & _ts;
          INT     _flag;

};




template <class AttrSom,class AttrArc> 
         class ElSubGraphe  : public ElSubGrapheSom<AttrSom,AttrArc>
{
     public :
          typedef ElSomIterator<AttrSom,AttrArc>  TSomIter;
          typedef ElArc<AttrSom,AttrArc>          TArc;
          typedef ElSom<AttrSom,AttrArc>          TSom;
          typedef AttrSom ASom;
          typedef AttrArc AArc;


          virtual bool   inS(TSom &)  {return true;}
          virtual bool   inA(TArc &)  {return true;} 
          virtual REAL   pds(TArc &) {return 1.0;} 
          virtual REAL   pds(TSom &) {return 0.0;} 
          virtual Pt2dr  pt(TSom &)
          {
                ELISE_ASSERT(false,"ElSubGraphe::pt");
                return Pt2dr();
          }

          bool inA(TSom &s1,TSom &s2)
          {
                TArc* a = s1.gr().arc_s1s2(s1,s2);
                return (a!=0) && (inA(*a));
          }

     private  :

};



template <class AttrSom,class AttrArc> class ElArc 
{
      public :
            typedef ElArc<AttrSom,AttrArc>          TArc;
            typedef ElSom<AttrSom,AttrArc>          TSom;
            typedef ElGraphe<AttrSom,AttrArc>       TGraphe;
            typedef ElArcIterator<AttrSom,AttrArc>  TArcIter;
            typedef AttrSom ASom;
            typedef AttrArc AArc;

            friend ClassFriend ElSom<AttrSom,AttrArc>;
            friend ClassFriend ElGraphe<AttrSom,AttrArc>;
            friend ClassFriend ElArcIterator<AttrSom,AttrArc>;

            TSom & s1()                   { return *_s1;}
            TSom & s2()                   { return *_s2;}
            AttrArc & attr()              { return _attr;}

	    bool IsAdj(const TSom & aSom)
	    {
                return (_s1==&aSom) || (_s2==&aSom);
	    }

            const TSom & s1()  const      { return *_s1;}
            const TSom & s2()  const      { return *_s2;}
            const AttrArc & attr() const  { return _attr;}

            inline ElArc  & arc_rec();
            inline TGraphe & gr();

            void    remove();

            bool  flag_kth(INT k) const         {return _flag.kth(k);  }
            void  flag_set_kth_true(INT k)      {_flag.set_kth_true(k);}
            void  flag_set_kth_false(INT k)     {_flag.set_kth_false(k);}
            void  flag_set_kth(INT k,bool val)  {_flag.set_kth(k,val);}
            void  flag_inv_kth(INT k)  {_flag.set_kth(k,!flag_kth(k));}


            void  sym_flag_set_kth_true(INT k)
            {
                  flag_set_kth_true(k);
                  arc_rec().flag_set_kth_true(k);
            }
            void  sym_flag_set_kth_false(INT k)
            {
                  flag_set_kth_false(k);
                  arc_rec().flag_set_kth_false(k);
            }

            void  sym_flag_set_kth(INT k,bool val)
            {
                  flag_set_kth(k,val);
                  arc_rec().flag_set_kth(k,val);
            }
            void sym_flag_inv_kth(INT k)
            {
                  sym_flag_set_kth(k,!flag_kth(k));
            }

            // Quand un choix arbitraire doit etre fait sur l'orientation
            bool IsOrientedNumCroissant() const
            {
                 return _s1->num() < _s2->num();
            }

      private  :

            ElArc(TSom & s1,TSom & s2,const AttrArc & attr):
                 _s1     (&s1),
                 _s2     (&s2),
                 _next   (0),
                 _attr   (attr)
            {
            }

            TSom *          _s1;
            TSom *          _s2;
            ElArc *      _next;
            AttrArc      _attr;
            ElTabFlag    _flag;
            ~ElArc(){}
};

template <class AttrSom,class AttrArc> class ElArcIterator
{
     public :
            typedef ElArc<AttrSom,AttrArc>        TArc;
            typedef ElSom<AttrSom,AttrArc>        TSom;
            typedef ElGraphe<AttrSom,AttrArc>     TGraphe;
            typedef ElSubGraphe<AttrSom,AttrArc>  TSubGraphe;
            typedef AttrSom ASom;
            typedef AttrArc AArc;

            friend ClassFriend  ElSom<AttrSom,AttrArc>;

            void operator ++ (INT) 
            {
                 if (_a)
                 {
                     do 
                     {
                        _a = _a->_next;
                     }
                     while 
                     (  
                              _a 
                           && (
                                    (!_sub_gr.inS(_a->s2()))
                                ||  (!_sub_gr.inA(*_a))  
                              )
                     );
                 }
            }
            bool   operator == (const ElArcIterator & it2) { return _a == it2._a;}
            bool   operator != (const ElArcIterator & it2) { return _a != it2._a;}
            TArc & operator *() {return  * _a;}
            TArc * operator ->() {return   _a;}

            bool  go_on() { return _a != 0;}

            

     private  :
            ElArcIterator(TSubGraphe & sub_gr,TArc * a) :
                    _sub_gr (sub_gr),
                    _a (a)
           {
           }

           void init_on()
           {
                while(_a && ((!_sub_gr.inA(*_a))||(!_sub_gr.inS(_a->s2()))))
                   _a = _a->_next;
           }
            
            TSubGraphe & _sub_gr;
            TArc *        _a;
};

template <class AttrSom,class AttrArc> class  ElSom 
{
       public :

            typedef ElArc<AttrSom,AttrArc>          TArc;
            typedef ElSom<AttrSom,AttrArc>          TSom;
            typedef ElGraphe<AttrSom,AttrArc>       TGraphe;
            typedef ElSubGraphe<AttrSom,AttrArc>    TSubGraphe;
            typedef ElArcIterator<AttrSom,AttrArc>  TArcIter;
            typedef ElSomIterator<AttrSom,AttrArc>  TSomIter;
            typedef AttrSom ASom;
            typedef AttrArc AArc;

            friend class ElArc<AttrSom,AttrArc>;
            friend class ElGraphe<AttrSom,AttrArc> ;
            friend class ElSubGraphe<AttrSom,AttrArc> ;
            friend class ElTabDyn<TSom>;
            friend class ElSomIterator<AttrSom,AttrArc>;

            INT nb_succ(TSubGraphe &);
            TArcIter begin(TSubGraphe & sub) 
            { 
                 TArcIter res(sub , (sub.inS(*this)?_succ:0));
                 res.init_on();
                 return res;
            }
			
			// => Erreur si aucun ou plusieur succ (pred)
			ElSom & uniq_succ(TSubGraphe & sub); 
			ElSom & uniq_pred(TSubGraphe & sub);
			// necessite graphe avec pt() et 1 seul succ et pred
			bool in_sect_angulaire(Pt2dr pt,TSubGraphe &); 

            TArcIter end(TSubGraphe & sub) { return TArcIter(sub,0);}

            void remove();
            AttrSom & attr() {return _attr;} 
            const AttrSom & attr() const {return _attr;} 
            INT num () const {return _num;}
            TGraphe  & gr() {return *_gr;}

            bool  flag_kth(INT k)  const        {return _flag.kth(k);  }
            void  flag_set_kth_true(INT k)      {_flag.set_kth_true(k);}
            void  flag_set_kth_false(INT k)     {_flag.set_kth_false(k);}
            void  flag_set_kth(INT k,bool val)  {_flag.set_kth(k,val);}
			
            ElSom() : _alive(false), _gr(0), _succ(0), _attr(), _num(), _flag() {}
            ~ElSom();
       private  :
            TArc * _remove_succ(TSom *);

            ElSom(TGraphe* gr,const AttrSom & attr,INT Num) ;

            bool         _alive;
            TGraphe *    _gr;
            TArc  *      _succ;
            AttrSom      _attr;
            INT          _num;
            ElTabFlag    _flag;
            bool really_in(TSubGraphe & sub){return _alive && sub.inS(*this);}
};

template <class  AttrSom,class AttrArc> class ElGraphe 
{
       public :

            typedef ElArc<AttrSom,AttrArc>          TArc;
            typedef ElSom<AttrSom,AttrArc>          TSom;
            typedef ElGraphe<AttrSom,AttrArc>       TGraphe;
            typedef ElSomIterator<AttrSom,AttrArc>  TSomIter;
            typedef ElArcIterator<AttrSom,AttrArc>  TArcIter;
            typedef ElSubGraphe<AttrSom,AttrArc>    TSubGraphe;
            typedef AttrSom ASom;
            typedef AttrArc AArc;

            friend class ElArc<AttrSom,AttrArc>;
            friend class ElSom<AttrSom,AttrArc>;
            friend class ElSomIterator<AttrSom,AttrArc>;

            TSom & new_som(const AttrSom & attr);

            ElGraphe() : _nb_som(0), _larc_free(0) {}

            TSomIter begin(TSubGraphe & sub) 
                     { TSomIter it(*this,sub,-1); it++; return it;}
            TSomIter end(TSubGraphe & sub)   
                     { return TSomIter(*this,sub,_tsom.nb());}

             INT   nb_som_phys() {return _tsom.nb();}
             INT   nb_som()      {return _nb_som;}
             INT   maj_num() {return _tsom.nb();}

             TArc * arc_s1s2(TSom & s1,TSom & s2);

             TArc & add_arc(TSom & s1,TSom &s2,const AttrArc & a1 ,const AttrArc & a2)
             {
                   ELISE_ASSERT(s1._gr==this&&s2._gr==this,"add_arc");
                   ELISE_ASSERT(s1._alive&&s2._alive,"KILLED SOM");
                   ELISE_ASSERT(! arc_s1s2(s1,s2),"add_arc : ARC EXIST");
                   ELISE_ASSERT(&s1!=&s2,"add_arc : same vertices");

                   _add_arc(s2,s1,a2);
                   TArc & aRes = _add_arc(s1,s2,a1);
                   OnNewArc(aRes);
                   return aRes;
             }
             TArc & add_arc(TSom & s1,TSom &s2,const AttrArc & a)
             {
                  return add_arc(s1,s2,a,a);
             }
             virtual ~ElGraphe();

             INT   alloc_flag_som() {return _flag_som_alloc.flag_alloc();}
             INT   alloc_flag_arc() {return _flag_arc_alloc.flag_alloc();}

             void    free_flag_som(INT k) { _flag_som_alloc.flag_free(k);}
             void    free_flag_arc(INT k) { _flag_arc_alloc.flag_free(k);}
            


       private  :

            virtual void OnNewSom(TSom &){}
            virtual void OnNewArc(TArc &){}
            virtual void OnKillSom(TSom &){}
            virtual void OnKillArc(TArc &){}

              void add_free(TArc * la){la->_next = _larc_free; _larc_free = la;}
              // void add_free(TArc * la){}
            TArc & _add_arc(TSom & s1,TSom &s2,const AttrArc & a1);
            static void kill_arc(TArc * la);

            INT             _nb_som;
            TArc *          _larc_free;
            ElTabDyn<TSom>  _tsom;
            ElFilo<INT>     _free_number;

            ElFlagAllocator   _flag_som_alloc;
            ElFlagAllocator   _flag_arc_alloc;
           
};

template <class AttrSom,class AttrArc> 
         ElGraphe<AttrSom,AttrArc> & ElArc<AttrSom,AttrArc>::gr()
{
     return *(_s1->_gr);
}


template <class AttrSom,class AttrArc> 
         ElArc<AttrSom,AttrArc> & ElArc<AttrSom,AttrArc>::arc_rec()
{
    return *(gr().arc_s1s2(*_s2,*_s1));
}

template <class AttrSom,class AttrArc> class ElSomIterator
{
     public :
            typedef ElArc<AttrSom,AttrArc>        TArc;
            typedef ElSom<AttrSom,AttrArc>        TSom;
            typedef ElGraphe<AttrSom,AttrArc>     TGraphe;
            typedef ElSubGraphe<AttrSom,AttrArc>  TSubGraphe;

            friend class ElGraphe<AttrSom,AttrArc> ;

            void operator ++ (INT) 
            {
                 do 
                      _k++ ;
                 while 
                 (
                        (_k<_gr.nb_som_phys())
                     && ( !(_gr._tsom[_k].really_in(_sub_gr)))
                 );
            }
            bool   operator == (const ElSomIterator & it2) { return _k == it2._k;}
            bool   operator != (const ElSomIterator & it2) { return _k != it2._k;}
            TSom & operator *() {return _gr._tsom[_k];}

            bool  go_on() { return _k < _gr.nb_som_phys();}
            

     private  :

            ElSomIterator(TGraphe & gr,TSubGraphe & sub_gr,INT k) :
                    _gr(gr),
                    _sub_gr (sub_gr),
                    _k (k)
           {
           }
            
            TGraphe & _gr;
            TSubGraphe & _sub_gr;
            INT  _k;

};


/*  Utilitaire  permettant de manipuler de maniere ensembliste des container de Type * gracea des flag
*/

     // Deux adaptateurs sur les sommets

template <class T1,class T2> inline bool ValFlag(ElSom<T1,T2> & aSom,int aFlagSom)
{
   return aSom.flag_kth(aFlagSom);
}
template <class T1,class T2> inline void  SetFlag(ElSom<T1,T2> & aSom,int aFlag,bool aVal)
{
   return aSom.flag_set_kth(aFlag,aVal);
}

     // Deux fonctions de manipulation

template <class TypeVal,class TypeCont> inline bool SetFlagAdd(TypeCont & aCont,TypeVal * aVal,int aFlag)
{
    if (! ValFlag(*aVal,aFlag))
    {
        SetFlag(*aVal,aFlag,true);
        aCont.push_back(aVal);
        return true;
    }
    return false;
}

template <class TypeCont> inline void FreeAllFlag(TypeCont & aCont,int aFlag)
{
    for (typename TypeCont::iterator it=aCont.begin() ; it!=aCont.end() ; it++)
    {
        SetFlag(**it,aFlag,false);
    }
}

template <class AttrSom,class AttrArc>
         class ElSubGrapheFlag  : public ElSubGraphe<AttrSom,AttrArc>
{
   public :
        ElSubGrapheFlag(int aFlagS,int aFlagA) :  // -1 => No Flag
             mFlagS (aFlagS),
             mFlagA (aFlagA)
        {
        }
        bool   inS(ElSom<AttrSom,AttrArc>  & aSom )  {return (mFlagS<0) ||  aSom.flag_kth(mFlagS);}
        bool   inA(ElArc<AttrSom,AttrArc>  & anArc)  {return (mFlagA<0) || anArc.flag_kth(mFlagA);}
   private :
        int mFlagS;
        int mFlagA;

};



#endif // _ELISE_GRAPHES_GRAPHE_H








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
