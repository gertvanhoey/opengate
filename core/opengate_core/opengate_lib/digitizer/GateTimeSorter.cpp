#include "GateTimeSorter.h"
#include "GateDigiCollection.h"
#include "GateDigiCollectionManager.h"
#include "GateHelpersDigitizer.h"
#include <G4Threading.hh>
#include <memory>
#include <utility>

GateTimeSorter::GateTimeSorter() {
  fNumThreads = G4Threading::GetNumberOfRunningWorkerThreads();
  fMaxGlobalTimePerThread =
      std::make_unique<std::atomic<double>[]>(fNumThreads);
}

void GateTimeSorter::Init(GateDigiCollection *input) {

  // Create an iterator for the input collection, tracking the GlobalTime of
  // digis to be able to sort them in time.
  fInputCollection = input;

  auto *manager = GateDigiCollectionManager::GetInstance();

  const auto name = fInputCollection->GetName();

  fBufferA = manager->NewDigiCollection(name + "_bufferA");
  fBufferA->InitDigiAttributesFromCopy(fInputCollection);
  fBufferA->SetSharedStorage(true);

  fBufferB = manager->NewDigiCollection(name + "_bufferA");
  fBufferB->InitDigiAttributesFromCopy(fInputCollection);
  fBufferB->SetSharedStorage(true);

  fSortedCollectionA = manager->NewDigiCollection(name + "_sortedA");
  fSortedCollectionA->InitDigiAttributesFromCopy(input);
  fSortedCollectionA->SetSharedStorage(true);
  fSortedIndicesA.reset(new TimeSortedIndices);

  fSortedCollectionB = manager->NewDigiCollection(name + "_sortedB");
  fSortedCollectionB->InitDigiAttributesFromCopy(input);
  fSortedCollectionB->SetSharedStorage(true);
  fSortedIndicesB.reset(new TimeSortedIndices);

  fOutputCollection = manager->NewDigiCollection(name + "_sortedOut");
  fOutputCollection->InitDigiAttributesFromCopy(fInputCollection);
  fOutputCollection->SetSharedStorage(true);

  fOutputIter = fOutputCollection->NewIterator();

  const auto attribute_names = input->GetDigiAttributeNames();

  fFillers[{input, fSortedCollectionA}] =
      std::make_unique<GateDigiAttributesFiller>(input, fSortedCollectionA,
                                                 attribute_names);
  fFillers[{input, fSortedCollectionB}] =
      std::make_unique<GateDigiAttributesFiller>(input, fSortedCollectionB,
                                                 attribute_names);

  // fFillers[{input, fBufferA}] = std::make_unique<GateDigiAttributesFiller>(
  //     input, fBufferA, attribute_names);
  // fFillers[{input, fBufferB}] = std::make_unique<GateDigiAttributesFiller>(
  //     input, fBufferB, attribute_names);
  // fFillers[{fBufferA, fSortedCollectionA}] =
  //     std::make_unique<GateDigiAttributesFiller>(fBufferA,
  //     fSortedCollectionA,
  //                                                attribute_names);
  // fFillers[{fBufferB, fSortedCollectionA}] =
  //     std::make_unique<GateDigiAttributesFiller>(fBufferB,
  //     fSortedCollectionA,
  //                                                attribute_names);
  // fFillers[{fBufferA, fSortedCollectionB}] =
  //     std::make_unique<GateDigiAttributesFiller>(fBufferA,
  //     fSortedCollectionB,
  //                                                attribute_names);
  // fFillers[{fBufferB, fSortedCollectionB}] =
  //     std::make_unique<GateDigiAttributesFiller>(fBufferB,
  //     fSortedCollectionB,
  //                                                attribute_names);

  fFillers[{fSortedCollectionA, fOutputCollection}] =
      std::make_unique<GateDigiAttributesFiller>(
          fSortedCollectionA, fOutputCollection, attribute_names);
  fFillers[{fSortedCollectionB, fOutputCollection}] =
      std::make_unique<GateDigiAttributesFiller>(
          fSortedCollectionB, fOutputCollection, attribute_names);

  fInitialized = true;
}

void GateTimeSorter::SetSortingWindow(double duration) {
  // The sorting window, specified in nanoseconds, is the difference in
  // GlobalTime between the oldest and newest digi being stored in the
  // TimeSortedStorage object, before they are transferred to the output
  // collection. Since the GlobalTime of incoming digis is not guaranteed to
  // increase monotonically, the sorting window must be large enough to ensure
  // that all digis can be properly time-sorted.
  // Non-monotonicity of GlobalTime can be caused by the time difference between
  // a particle's generation and its interaction with a detector. It can also be
  // caused by a DigitizerBlurringActor with blur_attribute "GlobalTime".

  if (fProcessingStarted) {
    Fatal("SetDelay() cannot be called after Process() has been called.");
  }
  fMinimumSortingWindow = duration;
  fSortingWindow = duration;
}

void GateTimeSorter::SetMaxSize(size_t maxSize) {
  if (fProcessingStarted) {
    Fatal("SetMaxSize() cannot be called after Process() has been called.");
  }
  fMaxSize = maxSize;
}

GateDigiCollection *GateTimeSorter::OutputCollection() const {
  return fOutputCollection;
}

std::unique_ptr<GateDigiAttributesFiller>
GateTimeSorter::CreateFiller(GateDigiCollection *destination) {
  // Creates a GateDigiAttributesFiller from the TimeSorter's output collection
  // into the given destination GateDigiCollection.

  if (!fInitialized) {
    Fatal("CreateFiller() called before Init().");
  }
  return std::make_unique<GateDigiAttributesFiller>(
      fOutputCollection, destination,
      fOutputCollection->GetDigiAttributeNames());
}

GateDigiCollection::Iterator &GateTimeSorter::OutputIterator() {
  // Provides access to the GateTimeSorter's output iterator.
  return fOutputIter;
}

void GateTimeSorter::Process() {
  // Processes all digis from the input collection by copying and sorting them
  // according to GlobalTime. Next, copies the oldest sorted digis to the output
  // collection.

  if (fFlushed) {
    Fatal("Process called after Flush(). The time sorter must not be used "
          "after it was flushed.");
  }
  fProcessingStarted = true;

  auto iter = fInputCollection->NewIterator();
  double *t;
  iter.TrackAttribute("GlobalTime", &t);

  // Iterate over the input collection and sort the digis.
  auto fillerIn = fFillers[{fInputCollection, fSortedCollectionA}].get();
  auto fillerOut = fFillers[{fSortedCollectionA, fOutputCollection}].get();
  iter.GoToBegin();
  while (!iter.IsAtEnd()) {
    const size_t digiIndex = fSortedCollectionA->GetSize();
    const double digiTime = *t;
    if (fMostRecentTimeDeparted.has_value() &&
        (digiTime < *fMostRecentTimeDeparted)) {
      // The digi is dropped, in order to be able to guarantee monotonous
      // GlobalTime in the output collection.
      ++fNumDroppedDigi;
      if (!fSortingWindowWarningIssued) {
        std::cout << "The digis in " << fInputCollection->GetName()
                  << " have non-monotonicities in the GlobalTime attribute "
                     "that exceed the sorting time ("
                  << fSortingWindow
                  << " ns). Please increase the sorting window to avoid "
                     "dropped digis\n";
        fSortingWindowWarningIssued = true;
      }
    } else {
      if (fNumThreads > 1) {
        const int tid = G4Threading::G4GetThreadId();
        const double currentMax = fMaxGlobalTimePerThread[tid].load();
        fMaxGlobalTimePerThread[tid].store(std::max(currentMax, digiTime));
      }
      // Copy the digi into the temporary digi collection.
      fillerIn->Fill(iter.fIndex);
      // Keep a time-sorted list of indices into the temporary digi collection.
      fSortedIndicesA->push({digiIndex, digiTime});

      // Keep track of the highest GlobalTime observed so far.
      if (!fMostRecentTimeArrived || (digiTime > *fMostRecentTimeArrived)) {
        fMostRecentTimeArrived = digiTime;
      }
      // Increase sorting window if needed.
      if (fNumThreads > 1) {
        auto [minIt, maxIt] = std::minmax_element(
            fMaxGlobalTimePerThread.get(),
            fMaxGlobalTimePerThread.get() + fNumThreads,
            [](const std::atomic<double> &a, const std::atomic<double> &b) {
              return a.load() < b.load();
            });
        double globalTimeSlowestThread = minIt->load();
        double globalTimeFastestThread = maxIt->load();
        const auto window = fMinimumSortingWindow + globalTimeFastestThread -
                            globalTimeSlowestThread;
        if (window > fSortingWindow) {
          fSortingWindow = window;
        }
      }
    }
    iter++;
  }

  // Copy the oldest digis from the sorted temporary storage into the output
  // collection. Continue as long as the newest digi is at least fSortingWindow
  // more recent than the oldest digi.
  while (!fSortedIndicesA->empty() &&
         (*fMostRecentTimeArrived - fSortedIndicesA->top().time >
          fSortingWindow)) {
    // Copy oldest digi into the output collection.
    fillerOut->Fill(fSortedIndicesA->top().index);
    // Keep track of the GlobalTime of the last digi that was copied.
    fMostRecentTimeDeparted = fSortedIndicesA->top().time;
    // Remove the time-sorted index of the digi.
    fSortedIndicesA->pop();
  }

  // The temporary digi collection keeps growing as more digis are processed.
  // The digis that have already been copied to the output must be removed once
  // in a while to limit memory usage.
  if (fSortedCollectionA->GetSize() > fMaxSize) {
    Prune();
  }
}

void GateTimeSorter::MarkOutputAsProcessed() {
  // Modifies the start of event index, to ensure that future use of the
  // iterator starts with the output digis that have not been processed yet.
  // Once in a while, the output collection is cleared to reduce memory
  // consumption.
  if (fOutputCollection->GetSize() <= fMaxSize) {
    fOutputCollection->SetBeginOfEventIndex(fOutputIter.fIndex);
  } else {
    fOutputCollection->Clear();
    fOutputIter.Reset();
  }
}

void GateTimeSorter::Flush() {
  // Copies all remaining sorted digis into the output collection.
  // This method is intended to be called once at the end of time sorting, when
  // it is known that no more digis will be processed from the input.
  // As a consequence, the sorting window does not have to be taken into account
  // while flushing.
  auto fillerOut = fFillers[{fSortedCollectionA, fOutputCollection}].get();
  while (fSortedIndicesA->size() > 0) {
    fillerOut->Fill(fSortedIndicesA->top().index);
    fSortedIndicesA->pop();
  }
  Prune();
  fFlushed = true;
  if (fNumDroppedDigi > 0) {
    std::cout << fNumDroppedDigi
              << " digis have been dropped while time-sorting. Please increase "
                 "the sorting time to a value higher than "
              << fMinimumSortingWindow << " ns\n";
  }
}

void GateTimeSorter::Prune() {
  // Frees memory that is currently occupied by digis that have already been
  // copied into the output collection:
  // 1. The digis that have not been copied to the output yet, are copied into
  // the second instance of TimeSortedStorage.
  // 2. The digi collection in the first instance of TimeSortedStorage is
  // cleared.
  // 3. The two instances of TimeSortedStorage are swapped.

  // Step 1
  std::cout << "GateTimeSorter::Prune()\n";
  GateDigiAttributesFiller transferFiller(
      fSortedCollectionA, fSortedCollectionB,
      fSortedCollectionA->GetDigiAttributeNames());
  // auto &sortedIndices = fCurrentStorage->sortedIndices;
  while (!fSortedIndicesA->empty()) {
    const auto timed_index = fSortedIndicesA->top();
    fSortedIndicesA->pop();
    const size_t digiIndex = fSortedCollectionB->GetSize();
    const double digiTime = timed_index.time;
    transferFiller.Fill(timed_index.index);
    fSortedIndicesB->push({digiIndex, digiTime});
  }
  // Step 2
  fSortedCollectionA->Clear();

  // Step 3
  std::swap(fSortedCollectionA, fSortedCollectionB);
  std::swap(fSortedIndicesA, fSortedIndicesB);
}
