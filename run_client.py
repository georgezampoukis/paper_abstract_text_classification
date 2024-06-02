from classification_api import ClassificationAPI




if __name__ == '__main__':

    # Set up Example Text
    example_text: str = r'Solitary lymphocytoma is a rare cutaneous manifestation of Lyme borreliosis that has been reported almost exclusively from Europe. This suggests that its etiologic \
                        agent may be absent or extremely rare on the North American continent. All three species of B. burgdorferi sensu lato known to be associated with human Lyme borreliosis \
                        (B. burgdorferi sensu stricto, B. garinii, and B. afzelii have been isolated in Europe, whereas only B. burgdorferi sensu stricto has been found \
                        in North America. This suggests that either B. garinii or B. afzelii might be the etiologic agent of borrelial lymphocytoma. To investigate this hypothesis \
                        we characterized five strains of B. burgdorferi sensu lato isolated from lymphocytoma lesions of patients residing in Slovenia. The methods used included: \
                        large restriction fragment pattern analysis of restriction enzyme MluI-digested genomic DNA, plasmid profiling, protein profiling, ribotyping using 5S, \
                        16S, and 23S rDNA probes, and polymerase chain reaction amplification of the rrf (5S)-rrl (23S) intergenic spacer region. Molecular subtyping showed that four of the \
                        five isolates belonged to the species B. afzelii; however, this species is the predominant patient isolate in Slovenia and, therefore, may not represent a preferential \
                        association with lymphocytoma. The fifth isolate appeared to be most closely related to the DN127 genomic group of organisms. \
                        Further characterization of the isolate revealed that it possessed a unique molecular "fingerprint." The results not only show that borrelial lymphocytoma \
                        can be caused by B. afzelii but also demonstrate an association with another genomic group of B. burgdorferi sensu lato that is present in North America as well.'
    
    # Set up Classifier API
    classifier: ClassificationAPI = ClassificationAPI()

    # Classify Text from Server
    server_response: dict = classifier.classify_text(example_text)

    print(server_response)