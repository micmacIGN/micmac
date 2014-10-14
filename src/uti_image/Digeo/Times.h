#ifndef __TIMES__
#define __TIMES__

#define __DEBUG_TIMES

class Times
{
public:
	virtual void clear() = 0;
	virtual void start() = 0;
	virtual double stop( const char *i_name ) = 0;
	virtual ~Times();
};

class NoTimes : public Times
{
public:
	inline void clear();
	inline void start();
	inline double stop( const char *i_name );
};

class MapTimes : public Times
{
private:
	class Record;

	typedef list<Record>::iterator ItRecord;

	class Record
	{
	public:
		string         mName;
		double         mTime;
		ElTimer        mTimer;
		list<ItRecord> mSubRecords;
		ItRecord       mFather;

		inline Record();
		double totalTime() const;
		void printTimes( const string &i_prefix, ostream &io_ostream ) const;
	};


	// return i_list.end() if a record of that name does not exist
	static list<ItRecord>::iterator getRecord( const string &i_name, list<ItRecord> &i_list );

	list<Record> mRecords;
	ItRecord     mCurrent;

public:
	MapTimes();
	inline void clear();
	void start();
	double stop( const char *i_name );
	inline double totalTime() const;
	void printTimes( const std::string &i_prefix=std::string(), std::ostream &io_ostream=std::cout ) const;
};

#include "Times.inline.h"

#endif // __TIMES__
