#ifndef __EXPRESSION__
#define __EXPRESSION__

#include <string>
#include <list>
#include <map>
#include <iostream>

#define __DEBUG_EXPRESSION

#define TO_CHILD(ChildType,x) (*(ChildType *)&(x))

class Expression
{
public:
	class Element
	{
	public:
		virtual bool isVariable() const;
		virtual ~Element();
		virtual void dump( std::ostream &io_stream=std::cout ) const = 0;
		virtual std::string toString() const = 0;

		// Element to child methods
		template <class T> inline T & operator ()();
		template <class T> inline const T & operator ()() const;

		static Element * duplicate( const Element &i_e );

		#ifdef __DEBUG_EXPRESSION
			typedef enum
			{
				MT_NEW, MT_DELETE
			} mt_type;

			typedef struct __memory_transaction
			{
				mt_type     type;
				Element *   object;
				std::string message;

				__memory_transaction( mt_type i_type, Element *i_object, const std::string &i_msg ):type(i_type), object(i_object), message(i_msg) {}
			} memory_transaction;

			void __allocating( const std::string &i_msg);
			void __deleting( const std::string &i_msg );
			void __dump_allocated();
			void __dump_transactions();

			static std::list<std::pair<Expression::Element*, std::string> > __allocated_elements;
			static std::list<memory_transaction> __memory_transactions;
		#endif
	};

	class Variable : public Element
	{
	private:
		std::string mName;
		int mMinWidth;
		char mFillCharacter;
		bool mIsValid;

	public:
		std::list<std::string> mUnusedElements;

		Variable( std::string i_string );

		// getters
		inline bool        isValid() const;
		inline std::string name() const;
		inline int         minWidth() const;
		inline char        fillCharacter() const;

		// Element's methods
		bool isVariable() const;
		std::string toString() const;

		void dump( std::ostream &io_stream=std::cout ) const;

		template <class T>
		std::string toString( const T &i_value );

		#ifdef __DEBUG_EXPRESSION
			Variable( const Variable &b );
			~Variable();
		#endif
	};

	class String : public Element
	{
	public:
		std::string mValue;
		
		inline String( const std::string &i_value );
		void dump( std::ostream &io_stream=std::cout ) const;
		std::string toString() const;

		#ifdef __DEBUG_EXPRESSION
			String( const String &b );
			~String();
		#endif
	};

	void set( const Expression &i_b );
	void set( const std::string &i_str );
	void clear();
	inline Expression( const Expression &i_str );
	inline Expression( const std::string &i_str=std::string() );
	inline Expression & operator =( const Expression &i_b );
	inline Expression & operator =( const std::string &i_b );
	inline ~Expression();

	// return the number of replacements made
	template <class T>
	int replace( const std::string &i_variableName, const T &i_value );

	template <class T>
	int replace( const std::map<std::string,T> &i_dico );

	bool hasVariable( const std::string &i_variableName ) const;
	bool hasVariables( const std::list<std::string> &i_variableNames ) const;

	inline bool isString() const;

	// value is set to
	template <class T>
	std::string value( const std::map<std::string,T> &i_dico, std::list<std::string> *o_unreplacedVariables=NULL ) const;

	void dump( std::ostream &io_stream=std::cout ) const;

	std::string toString() const;

	inline bool isValid() const;

private:
	void merge_subsequent_strings();

	std::list<Element*> mElements;
	bool                mIsValid;
};


//----------------------------------------------------------------------
// function related to class Expression
//----------------------------------------------------------------------

bool isInteger( const std::string &i_str );

std::ostream & operator <<( std::ostream &io_stream, const Expression::Element &i_element );


//----------------------------------------------------------------------

#include "Expression.inline.h"

#endif // __EXPRESSION__
