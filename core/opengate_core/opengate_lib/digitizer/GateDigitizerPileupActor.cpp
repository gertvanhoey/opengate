/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateDigitizerPileupActor.h"
#include "../GateHelpersDict.h"

GateDigitizerPileupActor::GateDigitizerPileupActor(py::dict &user_info)
    : GateVDigitizerWithOutputActor(user_info, false) {

  // Actions
  fActions.insert("EndOfEventAction");
}

GateDigitizerPileupActor::~GateDigitizerPileupActor() = default;

void GateDigitizerPileupActor::InitializeUserInfo(py::dict &user_info) {
  GateVDigitizerWithOutputActor::InitializeUserInfo(user_info);
  // blurring method
  fPileupTime = DictGetDouble(user_info, "pileup_time");
}

void GateDigitizerPileupActor::SetPileupTime(double pileupTime) {
  fPileupTime = pileupTime;
}

void GateDigitizerPileupActor::DigitInitialize(
    const std::vector<std::string> &attributes_not_in_filler) {
  auto a = attributes_not_in_filler;
  // ???
  GateVDigitizerWithOutputActor::DigitInitialize(a);

  CheckRequiredAttribute(fInputDigiCollection, "TotalEnergyDeposit");
  CheckRequiredAttribute(fInputDigiCollection, "GlobalTime");

  // set output pointers to the attributes needed for computation
  fOutputEdepAttribute =
      fOutputDigiCollection->GetDigiAttribute("TotalEnergyDeposit");
  fOutputGlobalTimeAttribute =
      fOutputDigiCollection->GetDigiAttribute("GlobalTime");

  // set input pointers to the attributes needed for computation
  auto &l = fThreadLocalData.Get();
  auto &lr = fThreadLocalVDigitizerData.Get();
  lr.fInputIter.TrackAttribute("TotalEnergyDeposit", &l.edep);
  lr.fInputIter.TrackAttribute("GlobalTime", &l.time);
}

void GateDigitizerPileupActor::BeginOfRunAction(const G4Run *run) {
  GateVDigitizerWithOutputActor::BeginOfRunAction(run);
}

void GateDigitizerPileupActor::EndOfEventAction(const G4Event * /*unused*/) {
  // fOutputEdepAttribute->Fill
  // fOutputGlobalTimeAttribute->Fill
}
