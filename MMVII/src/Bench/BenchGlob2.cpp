#include "include/MMVII_all.h"

namespace MMVII
{


template <class Type,class Compare, class TParam=DefaultParamHeap<Type> > class cHeap 
{
    public :

        cHeap(Compare inferior,size_t capa = 2) :
           mIsInferior    (inferior)
        {
		mElements.reserve(capa);
        }


        void MajOrAdd(const Type & aV)


    private :
        Compare           mIsInferior;
	std::vector<Type> mElements;

};


#if (0)

/*************************************************************/
/* SOME UTILS ON TAB                                         */
/*************************************************************/

#define HEAP_NO_INDEX -1

template <class Type,class Compare, class TParam=DefaultParamHeap<Type> > class ElHeap 
{
    public :

        ElHeap(Compare inferior,INT capa = 2) :
           _inferior    (inferior)
        {
		mEls.reserve(capa);
        }


        void MajOrAdd(const Type & aV)
        {
            if (TParam::Index(aV)==HEAP_NO_INDEX)
               push(aV);
            else
               MAJ(aV);
        }

        // Type & ValOfIndex(int aK) {return mEls[aK];}

        void MAJ(const Type & aV)
        {
           ReSync(TParam::Index(aV));
        }

        void Sortir(Type & v)
        {
            int i = TParam::Index(v);
            mEls[i] =  mEls[nb()-1] ; 
                                    /*----RI----*/ ResetIndex(i);
                                                    SetNoIndex(v); // SetNoIndex(mEls[nb()-1]);
            mEls.pop_back();
            ReSync(i);
        }

        void  push(const Type & v)
        {
                                 /*-------RI-----------*/ // TParam::SetIndex(v,nb()); // Ca va croitre
             mEls.push_back(v);  /*-------RI-----------*/    ResetIndex(nb()-1); //   ResetIndex(nb()-1);
             heap_down(nb()-1);
        }

        bool popPtr(Type * v) 
        {
             if (nb() <= 0) return false;
             if (v) {*v = mEls[0];  /*-------------RI-----------*/   SetNoIndex(*v);} //  TParam::SetIndex(v,HEAP_NO_INDEX);
             mEls[0] = mEls[nb()-1];  /*--RI-----------*/    ResetIndex(0);
	     mEls.pop_back();
             heap_up(0);
             return true;
        }


        bool pop(Type & v) {return popPtr(&v);}
        bool pop() {return popPtr(0);}


        Type * Lowest() 
        {
           if (empty()) return 0;
           return & mEls[0];
        }
        void clear() { mEls.clear();}
        INT nb() {return (int) mEls.size();}
        bool empty() {return mEls.empty();}

        const std::vector<Type> & Els() {return mEls;}

        Compare & Inferior() {return _inferior;}
        const Compare & Inferior() const {return _inferior;}

   private :

        Compare       _inferior;
	std::vector<Type> mEls;

        static INT  Fils(INT i) { return   (i+1)/2-1;}
        static INT  Pere1(INT i){ return   i*2+1;}
        static INT  Pere2(INT i){ return   i*2+2;}
        void ResetIndex(int aK)
        {
              TParam::SetIndex(mEls[aK],aK);
        }
        void SetNoIndex(Type & v)
        {
            TParam::SetIndex(v,HEAP_NO_INDEX);
        }


        Compare       _inferior;

        // do not exist
        ElHeap(const ElHeap<Type,Compare> &);

        void ReSync(int i)
        {
            if (i< nb())
            {
                if (!heap_down(i))
                   heap_up(i);
            }
        }

        void heap_up(INT KV)
        {
            for(;;)
            {
                INT ind_plus_petit = KV;

                INT p1 = Pere1(KV);
                if ((p1 < nb()) && _inferior(mEls[p1],mEls[ind_plus_petit]))
                    ind_plus_petit = p1;

                INT p2 = Pere2(KV);
                if ((p2 < nb()) && _inferior(mEls[p2],mEls[ind_plus_petit]))
                    ind_plus_petit = p2;

                if (ind_plus_petit == KV)
                   return;
                else
                {
                    ElSwap(mEls[ind_plus_petit],mEls[KV]);  /*-RI-*/  ResetIndex(ind_plus_petit);ResetIndex(KV);
                    KV = ind_plus_petit;
                }
            }
        }

        bool heap_down(INT KV)
        {
            bool SthDone = false;
            while ((KV>0)&& _inferior(mEls[KV],mEls[Fils(KV)]))
            {
                  ElSwap(mEls[KV],mEls[Fils(KV)]); /*-------RI------*/ ResetIndex(KV); ResetIndex(Fils(KV));
                  KV = Fils(KV);
                  SthDone = true;
            }
            return SthDone;
        }
};


/**  Class to use for non indexable heap */
template <class Type> class cDefaultParamHeap
{
     public :
        static void SetIndex(Type &,int i) {}
        static int  Index(const Type &) 
        {
             ELISE_ASSERT(false,"No DefaultParamHeap::Index");
             return HEAP_NO_INDEX;
        }
};


template <class Type,class Compare> class cTplKPluGrand
{

    public :
       cTplKPluGrand(Compare inferior,int aK) :
          mHeap (inferior,aK),
          mK    (aK)
       {
       }

       const std::vector<Type> & Els() {return mHeap.Els();}


      void push(const Type& aV)
      {
          if (mHeap.nb() < mK)
          {
               mHeap.push(aV);
          }
          else if (mHeap.Inferior()(*mHeap.Lowest(),aV))
          {
              mHeap.push(aV);
              mHeap.pop();
          }
      }
      void clear() { mHeap.clear();}
      void SetK(int aK)
      {
          while (mHeap.nb() > aK) mHeap.pop();
          mK = aK;
      }
      void ClearAndSetK(int aK)
      {
           mHeap.clear();
           mK = aK;
      }

    private :
          ElHeap<Type,Compare> mHeap ;
          int                  mK;

};


template <class TyVal,class TyPrio> class cTplPrioByOther
{
     public :
          cTplPrioByOther(const TyVal & aVal,const TyPrio & aPrio) :
               mVal (aVal),
               mPrio (aPrio)
          {
          }
          TyVal mVal;
          TyPrio mPrio;
};

template <class TyVal,class TyPrio>   class  cCmpSupPBO
{
    public :
       bool operator () (const cTplPrioByOther<TyVal,TyPrio> & aS1, const cTplPrioByOther<TyVal,TyPrio>  & aS2)
       {
           return aS1.mPrio > aS2.mPrio;
       }

};


#endif 


/*


template <class Type,class Compare> class ElHeap : private ElFifo<Type>
{
    public :

        ElHeap(Compare inferior,INT capa = 2) :
           ElFifo<Type> (capa),
           _inferior    (inferior)
        {
        }

        void  push(Type v)
        {
             pushlast(v);
             heap_down(_nb-1);
        }

        bool pop(Type & v)
        {
             if (_nb <= 0) return false;
             v = _tab[0];
             _tab[0] = _tab[_nb-1];
             _nb--;
             heap_up(0);
             return true;
        }
        void clear() { _nb = 0;}
        INT nb() {return _nb;}
        bool empty() {return nb() ==0;}

   private :

        static INT  Fils(INT i) { return   (i+1)/2-1;}
        static INT  Pere1(INT i){ return   i*2+1;}
        static INT  Pere2(INT i){ return   i*2+2;}


        Compare       _inferior;

        // do not exist
        ElHeap(const ElHeap<Type,Compare> &);

        void heap_up(INT KV)
        {
            for(;;)
            {
                INT ind_plus_petit = KV;

                INT p1 = Pere1(KV);
                if ((p1 < _nb) && _inferior(_tab[p1],_tab[ind_plus_petit]))
                    ind_plus_petit = p1;

                INT p2 = Pere2(KV);
                if ((p2 < _nb) && _inferior(_tab[p2],_tab[ind_plus_petit]))
                    ind_plus_petit = p2;

                if (ind_plus_petit == KV)
                   return;
                else
                {
                    ElSwap(_tab[ind_plus_petit],_tab[KV]);
                    KV = ind_plus_petit;
                }
            }
        }

        void heap_down(INT KV)
        {
            while ((KV>0)&& _inferior(_tab[KV],_tab[Fils(KV)]))
            {
                  ElSwap(_tab[KV],_tab[Fils(KV)]);
                  KV = Fils(KV);
            }
        }
};

template <class Type,class Compare> class ElHeap : private ElFifo<Type>
{
    public :

        ElHeap(Compare inferior,INT capa = 2) :
           ElFifo<Type> (capa),
           _inferior    (inferior)
        {
              pushlast(Type());
        }

        void  push(Type v)
        {
             pushlast(v);
             heap_down(_nb-1);
        }

        bool pop(Type & v)
        {
             if (_nb <= 1) return false;
             v = _tab[1];
             _tab[1] = _tab[_nb-1];
             _nb--;
             heap_up(1);
             return true;
        }
        void clear() { _nb = 1;}
        INT nb() {return _nb-1;}
        bool empty() {return nb() ==0;}

   private :

        static INT  Fils(INT i) { return   i/2;}
        static INT  Pere1(INT i){ return   i*2;}
        static INT  Pere2(INT i){ return   i*2+1;}


        Compare       _inferior;

        // do not exist
        ElHeap(const ElHeap<Type,Compare> &);

        void heap_up(INT KV)
        {
            for(;;)
            {
                INT ind_plus_petit = KV;

                INT p1 = Pere1(KV);
                if ((p1 < _nb) && _inferior(_tab[p1],_tab[ind_plus_petit]))
                    ind_plus_petit = p1;

                INT p2 = Pere2(KV);
                if ((p2 < _nb) && _inferior(_tab[p2],_tab[ind_plus_petit]))
                    ind_plus_petit = p2;

                if (ind_plus_petit == KV)
                   return;
                else
                {
                    ElSwap(_tab[ind_plus_petit],_tab[KV]);
                    KV = ind_plus_petit;
                }
            }
        }

        void heap_down(INT KV)
        {
            while ((KV>1)&& _inferior(_tab[KV],_tab[Fils(KV)]))
            {
                  ElSwap(_tab[KV],_tab[Fils(KV)]);
                  KV = Fils(KV);
            }
        }
};

*/





