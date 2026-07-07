#ifndef GateTimeSorter_h
#define GateTimeSorter_h

#include "GateDigiCollectionIterator.h"
#include <G4Threading.hh>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
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

  void OnEndOfEventAction(std::function<void(void)> work);
  void OnEndOfRunAction(std::function<void(void)> anyThreadWork,
                        std::function<void(void)> lastThreadWork);

  bool IsFirstUpstream() const;

  GateDigiCollection *OutputCollection() const;
  GateDigiCollection::Iterator &OutputIterator();
  void MarkOutputAsProcessed();

private:
  bool Ingest();
  void Process();
  void Flush();

  void IdentifyFastestThread();
  void MarkThreadAsFinished(int threadId);
  void Prune();

  double fMinimumSortingWindow{1000.0}; // nanoseconds
  size_t fMaxSize{100'000};             // digis

  // Threading

  G4Mutex fIngestionMutex;
  int fNumWorkingThreads{};
  std::atomic<double> fSortingWindow{1000.0}; // nanoseconds
  std::atomic<int> fFastestThread{};
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
  std::optional<double> fMostRecentTimeArrived;
  std::optional<double> fMostRecentTimeDeparted;

  // Most-upstream detection
  std::atomic<bool> fIsFirstUpstream{false};
  static std::atomic<GateTimeSorter *> sMostUpstreamInstance;

  // Thread convergence barrier (active on the most-upstream instance only,
  // in multi-threaded simulations).
  // When fSortedCollectionA first reaches kBarrierActivationThreshold digis,
  // the GlobalTime divergence between the fastest and slowest thread is
  // recorded. From that point on, threads whose GlobalTime has reached the
  // current barrier target spin-wait until all other threads have caught up.
  // The target then advances by the recorded divergence, repeating the barrier.
  static constexpr size_t kBarrierActivationThreshold = 50'000;
  std::atomic<bool> fBarrierSetupClaimed{
      false};                              // won by the thread that does setup
  std::atomic<bool> fBarrierActive{false}; // true once setup is complete
  std::atomic<bool> fBarrierBypassed{
      false}; // set in OnEndOfRunAction to unblock spinners
  double fRecordedDivergence{
      0.0}; // written once before fBarrierActive becomes true
  std::atomic<double> fBarrierTarget{0.0}; // current barrier GlobalTime target
  std::atomic<int> fThreadsAtBarrier{
      0}; // threads currently waiting at the barrier
  std::atomic<int> fBarrierGeneration{
      0}; // incremented each time the barrier is released
  std::atomic<size_t> fSortedCollectionASize{
      0}; // updated in Process(); avoids racy GetSize() reads
};

#endif // GateTimeSorter_h
