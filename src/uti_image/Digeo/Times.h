#ifndef __TIMES__
#define __TIMES__

#include "debug.h"

#ifdef NO_ELISE
	#include <list>
	#include <string>
	#include <iostream>

	class Timer
	{
	public:
		double mRecordedTime;

		Timer();
		void reinit();
		double uval();
	};
#else
	#include "StdAfx.h"

	typedef ElTimer Timer;
#endif

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
	void clear();
	void start();
	double stop( const char *i_name );
};

class MapTimes : public Times
{
private:
	class Record;
	typedef std::list<Record>::iterator ItRecord;

	class Record
	{
	public:
		std::string         mName;
		double              mTime;
		Timer               mTimer;
		std::list<ItRecord> mSubRecords;
		ItRecord            mFather;

		Record();
		double totalTime() const;
		void printTimes( const std::string &i_prefix, std::ostream &io_ostream ) const;
		bool hasRecord( const std::string &i_name ) const;
	};


	// return i_list.end() if a record of that name does not exist
	static std::list<ItRecord>::iterator getRecord( const std::string &i_name, std::list<ItRecord> &i_list );

	std::list<Record> mRecords;
	ItRecord          mCurrent;

public:
	MapTimes();
	void clear();
	void start();
	double stop( const char *i_name );
	double totalTime() const;
	void printTimes( const std::string &i_prefix=std::string(), std::ostream &io_ostream=std::cout ) const;
	double getRecordTime( const std::string &aName ) const;
};

double getTime();

#include "Times.inline.h"

#endif // __TIMES__
