#ifndef GateTimeSorter_h
#define GateTimeSorter_h

#include "GateDigiCollectionIterator.h"
#include <G4Threading.hh>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>

class GateDigiCollection;
class GateDigiAttributesFiller;

class GateTimeSorter {
public:
  GateTimeSorter(const std::string &name);

  void Init(GateDigiCollection *input);

  void SetSortingWindow(double duration);
  void SetMaxSize(size_t size);
  void SetBufferThreadSyncThreshold(size_t size);
  void SetThreadSyncEnabled(bool enabled);

  void OnEndOfEventAction(std::function<void(void)> work);
  void OnEndOfRunAction(std::function<void(void)> anyThreadWork,
                        std::function<void(void)> lastThreadWork);

  GateDigiCollection *OutputCollection() const;
  GateDigiCollection::Iterator &OutputIterator();
  void MarkOutputAsProcessed();

private:
  bool Ingest();
  void Process();
  void Flush();

  bool IsFirstUpstream() const;
  void MarkThreadAsFinished(int threadId);
  void Prune();

  double fMinimumSortingWindow{1000.0}; // nanoseconds
  size_t fMaxSize{100'000};             // digis

  // Threading

  G4Mutex fIngestionMutex;
  int fNumWorkingThreads{};
  std::atomic<double> fSortingWindow{1000.0}; // nanoseconds
  std::atomic<int> fNumActiveWorkingThreads{};
  std::atomic<bool> fProcessingOngoing{};
  std::atomic<int> fNumIngestions{};

  struct alignas(64) PaddedAtomicDouble {
    // Pad atomic<double> to one cache line (64 bytes) to prevent false sharing
    // when N threads each write to their own element in a tight loop.
    std::atomic<double> value{};
  };
  std::unique_ptr<PaddedAtomicDouble[]> fMaxGlobalTimePerThread;

  // Digi storage, sorting and copying

  struct TimedDigiIndex {
    size_t index;
    double time;

    bool operator>(const TimedDigiIndex &other) const {
      return time > other.time;
    }
  };

  typedef std::priority_queue<TimedDigiIndex, std::vector<TimedDigiIndex>,
                              std::greater<TimedDigiIndex>>
      TimeSortedIndices;

  GateDigiCollection *fInputCollection;

  GateDigiCollection *fIngestionBufferA;
  GateDigiCollection *fIngestionBufferB;

  GateDigiCollection *fSortedCollectionA;
  std::unique_ptr<TimeSortedIndices> fSortedIndicesA;
  GateDigiCollection *fSortedCollectionB;
  std::unique_ptr<TimeSortedIndices> fSortedIndicesB;

  GateDigiCollection *fOutputCollection;
  GateDigiCollectionIterator fOutputIter;

  std::map<std::pair<GateDigiCollection *, GateDigiCollection *>,
           std::unique_ptr<GateDigiAttributesFiller>>
      fFillers;

  // GateTimeSorter internal state

  std::string fName{};
  bool fInitialized{false};
  bool fProcessingStarted{false};
  bool fFlushed{false};
  bool fSortingWindowWarningIssued{false};
  size_t fNumDroppedDigi{};
  size_t fNumDigi{};
  double fMaxDropDelta{};
  std::optional<double> fFirstGlobalTime;
  std::optional<double> fMostRecentTimeArrived;
  std::optional<double> fMostRecentTimeDeparted;

  // Most-upstream detection
  std::atomic<bool> fIsFirstUpstream{false};
  static std::atomic<GateTimeSorter *> sMostUpstreamInstance;

  // Barrier for thread synchronization
  size_t fBarrierActivationThreshold{50'000};
  bool fThreadSyncEnabled{true};
  std::atomic<bool> fBarrierSetupClaimed{false};
  std::atomic<bool> fBarrierSetupComplete{false};
  std::atomic<bool> fBarrierBypassed{false};
  double fRecordedGlobalTimeInterval{0.0};
  std::atomic<double> fBarrierGlobalTimeTarget{0.0};
  std::atomic<int> fNumThreadsAtBarrier{0};
  std::atomic<int> fBarrierGeneration{0};
  std::atomic<size_t> fSortedIndicesSize{0};
  std::mutex fBarrierConditionVariableMutex;
  std::condition_variable fBarrierConditionVariable;
};

#endif // GateTimeSorter_h
