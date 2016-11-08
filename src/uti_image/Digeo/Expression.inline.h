// this file should be included by Expression.h solely

#include <sstream>
#include <iomanip>

//----------------------------------------------------------------------
// methods of class Expression::Variable
//----------------------------------------------------------------------

bool Expression::Variable::isValid() const { return mIsValid; }

std::string Expression::Variable::name() const { return mName; }

int Expression::Variable::minWidth() const { return mMinWidth; }

char Expression::Variable::fillCharacter() const { return mFillCharacter; }

template <class T>
std::string Expression::Variable::toString( const T &i_value )
{
	std::stringstream ss;
	ss << std::setw(mMinWidth) << std::setfill(mFillCharacter) << i_value;
	return ss.str();
}

//----------------------------------------------------------------------
// methods of class Expression::String
//----------------------------------------------------------------------

Expression::String::String( const std::string &i_value ):mValue(i_value)
{
	#ifdef __DEBUG_EXPRESSION
		__allocating("Expression::String::String(const string &)");
	#endif
}


//----------------------------------------------------------------------
// methods of class Expression
//----------------------------------------------------------------------

Expression::Expression( const Expression &i_b ){ set(i_b); }

Expression::Expression( const std::string &i_str ){ set(i_str); }

Expression & Expression::operator =( const Expression &i_b ){ set(i_b); return *this; }

Expression & Expression::operator =( const std::string &i_str ){ set(i_str); return *this; }

Expression::~Expression(){ clear(); }

bool Expression::isValid() const { return mIsValid; }

bool Expression::isString() const { return ( mElements.size()==1 && !( **mElements.begin() ).isVariable() ); }

template <class T>
int Expression::replace( const std::string &i_variableName, const T &i_value )
{
	int nbReplaced = 0;
	std::list<Element*>::iterator itElement = mElements.begin();
	while ( itElement!=mElements.end() )
	{
		if ( (*itElement)->isVariable() )
		{
			Variable *var = (Variable*)*itElement;
			if ( var->name()==i_variableName )
			{
				// a variable has been found, replace the variable by a string
				String *newStr = new String( var->toString<T>(i_value) );
				delete var;
				*itElement = newStr;
				nbReplaced++;
			}
		}
		itElement++;
	}
	if ( nbReplaced!=0 ) merge_subsequent_strings();
	return nbReplaced;
}

template <class T>
int Expression::replace( const std::map<std::string,T> &i_dico )
{
	int nbReplaced = 0;
	typename std::map<std::string,T>::const_iterator it = i_dico.begin();
	while ( it!=i_dico.end() )
	{
		nbReplaced += replace<T>( it->first, it->second );
		it++;
	}
	return nbReplaced;
}

template <class T>
std::string Expression::value( const std::map<std::string,T> &i_dico, std::list<std::string> *o_unreplacedVariables ) const
{
	Expression p = *this;
	p.replace( i_dico );

	if ( !p.isString() )
	{
		if ( o_unreplacedVariables!=NULL )
		{
			o_unreplacedVariables->clear();
			std::list<Element*>::const_iterator it = p.mElements.begin();
			while ( it!=p.mElements.end() )
			{
				if ( (**it).isVariable() ) o_unreplacedVariables->push_back( TO_CHILD(Variable,**it).name() );
				it++;
			}
		}
	}

	return p.toString();
}
